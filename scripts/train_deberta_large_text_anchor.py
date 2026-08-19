"""Second-encoder replication, Step 1 (docs/PREREG_DeBERTa-v3-large.md):
fine-tuned text foundation with `microsoft/deberta-v3-large` in place of
`roberta-base`, mirroring `scripts/train_context_text.py --loss ce`'s Phase T
recipe as closely as the encoder swap allows. Does NOT modify
train_context_text.py (that script stays the frozen RoBERTa-base Phase T
anchor) -- this is a standalone parallel script, per this project's existing
convention of one script per foundation-rebuild phase.

Recipe (identical to Phase T's plain-CE recipe except where the encoder
swap strictly requires a change -- see docs/PREREG_DeBERTa-v3-large.md for
the two documented deviations, LoRA target modules and pooling-layer
correction): context k=8, max_length=256, masked-mean pooling of
last_hidden_state, plain CE, AdamW lr=2e-4, linear warmup 10%, max 10
epochs, early-stop patience 3 on val macro F1, bf16, batch by utterance
(32), grad_clip=1.0.

Usage:
    uv run python -m scripts.train_deberta_large_text_anchor --seed 42
    uv run python -m scripts.train_deberta_large_text_anchor --seed 1337 --run_name deberta_large_text_anchor_seed1337
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
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

NUM_CLASSES = len(EMOTION_LABELS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "microsoft/deberta-v3-large"
# DeBERTa-v3's disentangled-attention analog of RoBERTa's [query, value]
# (docs/PREREG_DeBERTa-v3-large.md) -- same role (query/value projections
# only), DeBERTa's actual module names.
LORA_TARGET_MODULES = ["query_proj", "value_proj"]

# Phase T's plain-CE recipe, unchanged (docs/RECIPE.md).
LR = 2e-4
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
WARMUP_FRACTION = 0.10
BATCH_SIZE = 32
GRAD_CLIP = 1.0
DEFAULT_K = 8
DEFAULT_MAX_LENGTH = 256


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
def collect_logits(model: ContextTextClassifier, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_split_metrics(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=-1).tolist()
    labels_list = labels.tolist()
    weighted_f1 = f1_score(labels_list, preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels_list, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels_list, preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    return weighted_f1, macro_f1, per_class_f1, preds, labels_list


def linear_warmup_schedule(optimizer: AdamW, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remaining = num_training_steps - step
        return max(0.0, remaining / max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def train(seed: int, run_dir: Path, k: int = DEFAULT_K, max_length: int = DEFAULT_MAX_LENGTH) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    loaders = build_dataloaders(k, max_length, tokenizer)
    model = ContextTextClassifier(
        num_classes=NUM_CLASSES, model_name=MODEL_NAME, lora_target_modules=LORA_TARGET_MODULES
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR)

    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    scheduler = linear_warmup_schedule(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

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

        val_logits, val_labels = collect_logits(model, loaders["dev"], device)
        val_weighted_f1, val_macro_f1, val_per_class_f1, _, _ = compute_split_metrics(val_logits, val_labels)

        train_loss = total_loss / max(n_batches, 1)
        per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, val_per_class_f1)}

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

    test_logits, test_labels = collect_logits(model, loaders["test"], device)
    test_weighted_f1, test_macro_f1, test_per_class_f1, test_preds, test_labels_list = compute_split_metrics(
        test_logits, test_labels
    )
    test_accuracy = sum(int(p == l) for p, l in zip(test_preds, test_labels_list)) / len(test_labels_list)

    report = evaluate_and_report(test_labels_list, test_preds, EMOTION_LABELS, run_dir)
    report.update(
        {
            "seed": seed,
            "model_name": MODEL_NAME,
            "lora_target_modules": LORA_TARGET_MODULES,
            "k": k,
            "max_length": max_length,
            "loss_kind": "ce",
            "trainable_params": model.trainable_params,
            "total_params": model.total_params,
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_accuracy,
            "test_per_class_f1": {l: float(f1) for l, f1 in zip(EMOTION_LABELS, test_per_class_f1)},
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": checkpoint["epoch"],
            "num_epochs_run": len(epoch_times),
            "epoch_times_sec": epoch_times,
            "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
            "history": history,
            "all_7_classes_nonzero": bool(all(f1 > 0 for f1 in test_per_class_f1)),
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

    run_name = args.run_name or f"deberta_large_text_anchor_seed{args.seed}"
    run_dir = PROJECT_ROOT / "outputs" / run_name

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {run_name} already has metrics.json", flush=True)
        report = json.loads(metrics_path.read_text())
    else:
        report = train(args.seed, run_dir, k=args.k, max_length=args.max_length)

    print(
        f"[done] run_name={run_name} test_weighted_f1={report['test_weighted_f1']:.4f} "
        f"test_macro_f1={report['test_macro_f1']:.4f} test_accuracy={report['test_accuracy']:.4f} "
        f"all_7_classes_nonzero={report['all_7_classes_nonzero']} "
        f"best_epoch={report['best_epoch']} avg_epoch_time_sec={report['avg_epoch_time_sec']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
