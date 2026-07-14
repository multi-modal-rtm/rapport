"""Phase T: contextual text encoder training (LoRA-tuned RoBERTa, CE + logit
adjustment). Standalone script, not the Hydra `rapport.__main__` pipeline --
this phase is a text-only foundation rebuild, predating any GNN/fusion
integration (see docs/RECIPE.md's Phase T scope note). Batches by
utterance, no dialogue-level recurrent state.

Usage:
    uv run python scripts/train_context_text.py --seed 42
    uv run python scripts/train_context_text.py --seed 42 --k 4 --run_name context_text_k4_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Must be set before numpy/torch/sklearn import -- see rapport/__main__.py's
# identical guard and docs/DIAGNOSIS.md (Recalibration Step 1) for why.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rapport.data.constants import EMOTION_LABELS
from rapport.data.context_text import ContextTextCollator, MELDContextTextDataset
from rapport.eval.report import evaluate_and_report
from rapport.models.text_classifier import ContextTextClassifier
from rapport.repro import snapshot_environment
from rapport.seed import set_seed
from rapport.training.losses import LogitAdjustedLoss, compute_class_priors

NUM_CLASSES = len(EMOTION_LABELS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LR = 2e-4
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
WARMUP_FRACTION = 0.10
BATCH_SIZE = 32
LOGIT_ADJUSTMENT_TAU = 1.0
GRAD_CLIP = 1.0
DEFAULT_K = 8
DEFAULT_MAX_LENGTH = 256

# Pre-registered gate (see phase instructions / docs/RECIPE.md Phase T section).
GATE_WEIGHTED_F1 = 0.60
GATE_LOWER_INVESTIGATE = 0.57


def build_dataloaders(k: int, max_length: int, tokenizer, num_workers: int = 4) -> dict[str, DataLoader]:
    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    collate = ContextTextCollator(pad_token_id=tokenizer.pad_token_id)
    loaders = {}
    for split in ("train", "dev", "test"):
        dataset = MELDContextTextDataset(processed_dir / f"{split}.parquet", tokenizer, k=k, max_length=max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            collate_fn=collate,
            num_workers=num_workers,
        )
    return loaders


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_split(model: ContextTextClassifier, loader: DataLoader, device: torch.device):
    """Raw (unadjusted) logits for prediction -- logit adjustment is a
    training-time-only correction (docs/RECIPE.md); eval never applies it.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return weighted_f1, macro_f1, all_labels, all_preds


def linear_warmup_schedule(optimizer: AdamW, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remaining = num_training_steps - step
        return max(0.0, remaining / max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def train(seed: int, k: int, max_length: int, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    loaders = build_dataloaders(k, max_length, tokenizer)
    model = ContextTextClassifier(num_classes=NUM_CLASSES).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR)

    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    scheduler = linear_warmup_schedule(optimizer, warmup_steps, total_steps)

    priors = compute_class_priors(loaders["train"].dataset.index_df["label"], NUM_CLASSES).to(device)
    criterion = LogitAdjustedLoss(priors, tau=LOGIT_ADJUSTMENT_TAU, ignore_index=-1)

    snapshot_environment(run_dir)
    ckpt_path = run_dir / "best_model.pt"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    epoch_times: list[float] = []
    history: list[dict] = []

    for epoch in range(MAX_EPOCHS):
        start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        val_weighted_f1, val_macro_f1, val_labels, val_preds = evaluate_split(model, loaders["dev"], device)
        per_class_f1 = f1_score(
            val_labels, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
        )
        train_loss = total_loss / max(n_batches, 1)
        per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, per_class_f1)}

        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} val_weighted_f1={val_weighted_f1:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} time={epoch_time:.1f}s lr={scheduler.get_last_lr()[0]:.2e} "
            f"per_class_f1={per_class_str}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_weighted_f1": val_weighted_f1,
                "val_macro_f1": val_macro_f1,
                "val_per_class_f1": per_class_str,
                "epoch_time_sec": epoch_time,
            }
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_macro_f1": val_macro_f1},
                ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"[trainer] early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})", flush=True)
                break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_weighted_f1, test_macro_f1, test_labels, test_preds = evaluate_split(model, loaders["test"], device)
    test_accuracy = sum(int(p == l) for p, l in zip(test_preds, test_labels)) / len(test_labels)

    report = evaluate_and_report(test_labels, test_preds, EMOTION_LABELS, run_dir)
    per_class_test_f1 = f1_score(
        test_labels, test_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    report.update(
        {
            "seed": seed,
            "k": k,
            "max_length": max_length,
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_accuracy,
            "test_per_class_f1": {l: float(f1) for l, f1 in zip(EMOTION_LABELS, per_class_test_f1)},
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": checkpoint["epoch"],
            "num_epochs_run": len(epoch_times),
            "epoch_times_sec": epoch_times,
            "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
            "history": history,
            "all_7_classes_nonzero": bool(all(f1 > 0 for f1 in per_class_test_f1)),
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or f"context_text_seed{args.seed}"
    run_dir = PROJECT_ROOT / "outputs" / run_name

    report = train(args.seed, args.k, args.max_length, run_dir)
    print(
        f"[done] run_name={run_name} test_weighted_f1={report['test_weighted_f1']:.4f} "
        f"test_macro_f1={report['test_macro_f1']:.4f} test_accuracy={report['test_accuracy']:.4f} "
        f"all_7_classes_nonzero={report['all_7_classes_nonzero']} "
        f"best_epoch={report['best_epoch']} avg_epoch_time_sec={report['avg_epoch_time_sec']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
