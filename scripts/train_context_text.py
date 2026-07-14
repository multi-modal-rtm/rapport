"""Phase T: contextual text encoder training (LoRA-tuned RoBERTa). Standalone
script, not the Hydra `rapport.__main__` pipeline -- this phase is a
text-only foundation rebuild, predating any GNN/fusion integration (see
docs/RECIPE.md's Phase T scope note). Batches by utterance, no
dialogue-level recurrent state.

`--loss la` (default): Menon et al. 2021's training-time logit-adjustment
variant -- CE over (logits - tau*log(prior)) during training, raw logits at
eval. This was the original Phase T recipe; see docs/PHASE_T.md /
docs/PHASE_T_DIAGNOSIS.md for why it failed to rescue fear/disgust.

`--loss ce`: plain cross-entropy training, no adjustment baked into the
loss at all (Step 4's approved amendment, docs/PHASE_T_DIAGNOSIS.md). Every
run, regardless of `--loss`, ALSO evaluates a post-hoc (inference-time-only)
logit-adjustment sweep after training completes -- this never touches
training and is pure instrumentation/candidate-selection.

Usage:
    uv run python scripts/train_context_text.py --seed 42 --loss la
    uv run python scripts/train_context_text.py --seed 42 --loss ce --run_name context_text_plain_ce_seed42
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
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rapport.data.constants import EMOTION_LABELS
from rapport.data.context_text import ContextTextCollator, MELDContextTextDataset
from rapport.eval.rank_diagnostics import (
    posthoc_adjustment_sweep,
    predicted_vs_true_histogram,
    rank_of_true_class_stats,
    top1_confusion_for_classes,
)
from rapport.eval.report import evaluate_and_report
from rapport.models.text_classifier import ContextTextClassifier
from rapport.repro import snapshot_environment
from rapport.seed import set_seed
from rapport.training.losses import LogitAdjustedLoss, compute_class_priors

NUM_CLASSES = len(EMOTION_LABELS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RARE_CLASSES = ["fear", "disgust"]

LR = 2e-4
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
WARMUP_FRACTION = 0.10
BATCH_SIZE = 32
LOGIT_ADJUSTMENT_TAU = 1.0  # train-time tau, only used when --loss la
GRAD_CLIP = 1.0
DEFAULT_K = 8
DEFAULT_MAX_LENGTH = 256

# Post-hoc (inference-time-only) candidate-selection rule, pre-registered
# with Step 4's sign-off (docs/PHASE_T_DIAGNOSIS.md): highest val macro F1
# subject to all-7-nonzero AND val weighted F1 >= VAL_CANDIDATE_WEIGHTED_F1_FLOOR.
POSTHOC_SWEEP_TAUS = [0.25, 0.5, 0.75, 1.0, 1.25]
VAL_CANDIDATE_WEIGHTED_F1_FLOOR = 0.58
# Amended TEST gate (was 0.60; 0.01 is inside seed std, structural class
# death is not -- docs/PHASE_T_DIAGNOSIS.md Step 4 sign-off).
TEST_GATE_WEIGHTED_F1_FLOOR = 0.59


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
    """Raw (unadjusted) logits -- logit adjustment, if any, is a training-time
    correction (docs/RECIPE.md); this never applies it. Any inference-time
    adjustment is a separate, explicit, pure-instrumentation step on top of
    these raw logits (`posthoc_adjustment_sweep`).
    """
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_split_metrics(logits: torch.Tensor, labels: torch.Tensor, log_priors: torch.Tensor | None = None, tau: float = 0.0):
    adjusted = logits - tau * log_priors if log_priors is not None and tau != 0.0 else logits
    preds = adjusted.argmax(dim=-1).tolist()
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


def train(seed: int, k: int, max_length: int, loss_kind: str, run_dir: Path, posthoc_tau: float | None = None) -> dict:
    assert loss_kind in ("la", "ce"), f"unknown --loss {loss_kind!r}, expected 'la' or 'ce'"
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
    log_priors = torch.log(priors)

    if loss_kind == "la":
        criterion = LogitAdjustedLoss(priors, tau=LOGIT_ADJUSTMENT_TAU, ignore_index=-1)
    else:
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

        # One pass of val logits, reused for both the raw (selection) metrics
        # and the tau=1-adjusted (observation-only) metrics -- instrumentation
        # requested alongside Step 4's sign-off, not a recipe change.
        val_logits, val_labels = collect_logits(model, loaders["dev"], device)
        val_weighted_f1, val_macro_f1, val_per_class_f1, _, _ = compute_split_metrics(val_logits, val_labels)
        adj_weighted_f1, adj_macro_f1, adj_per_class_f1, _, _ = compute_split_metrics(
            val_logits, val_labels, log_priors=log_priors.cpu(), tau=1.0
        )

        train_loss = total_loss / max(n_batches, 1)
        per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, val_per_class_f1)}
        adj_per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, adj_per_class_f1)}

        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} val_weighted_f1={val_weighted_f1:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} time={epoch_time:.1f}s lr={scheduler.get_last_lr()[0]:.2e}\n"
            f"    raw       per_class_f1={per_class_str}\n"
            f"    tau=1 adj per_class_f1={adj_per_class_str} (adj_weighted_f1={adj_weighted_f1:.4f} adj_macro_f1={adj_macro_f1:.4f})",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_weighted_f1": val_weighted_f1,
                "val_macro_f1": val_macro_f1,
                "val_per_class_f1": per_class_str,
                "val_tau1_adjusted_weighted_f1": adj_weighted_f1,
                "val_tau1_adjusted_macro_f1": adj_macro_f1,
                "val_tau1_adjusted_per_class_f1": adj_per_class_str,
                "epoch_time_sec": epoch_time,
            }
        )

        # Checkpoint selection and early stopping are always on RAW val macro
        # F1 -- the tau=1-adjusted numbers above are observation-only, never
        # used for selection (that would silently change the recipe).
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
            "k": k,
            "max_length": max_length,
            "loss_kind": loss_kind,
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

    # Post-training diagnostics on VAL, at the selected checkpoint: rank of
    # true class + top-1 confusion for the rare classes (raw logits), and the
    # post-hoc adjustment sweep used for candidate selection below.
    val_logits, val_labels = collect_logits(model, loaders["dev"], device)
    val_rank_stats = rank_of_true_class_stats(val_logits, val_labels, EMOTION_LABELS)
    val_confusion = top1_confusion_for_classes(val_logits, val_labels, EMOTION_LABELS, RARE_CLASSES)
    val_sweep = posthoc_adjustment_sweep(val_logits, val_labels, log_priors.cpu(), POSTHOC_SWEEP_TAUS, EMOTION_LABELS)

    if posthoc_tau is not None:
        # Frozen recipe value (selected once on seed 42's val split, per this
        # project's established tau-freezing convention -- docs/RECIPE.md's
        # Amendment 2). Not re-selected per seed.
        chosen_tau = posthoc_tau
    else:
        candidates = [
            row for row in val_sweep if row["all_classes_nonzero"] and row["weighted_f1"] >= VAL_CANDIDATE_WEIGHTED_F1_FLOOR
        ]
        chosen_tau = max(candidates, key=lambda r: r["macro_f1"])["tau_eval"] if candidates else None

    test_posthoc = None
    if chosen_tau is not None:
        test_sweep_result = posthoc_adjustment_sweep(test_logits, test_labels, log_priors.cpu(), [chosen_tau], EMOTION_LABELS)[0]
        test_posthoc = test_sweep_result
        test_posthoc["passes_amended_gate"] = bool(
            test_sweep_result["all_classes_nonzero"] and test_sweep_result["weighted_f1"] >= TEST_GATE_WEIGHTED_F1_FLOOR
        )

    report.update(
        {
            "val_predicted_vs_true_histogram": predicted_vs_true_histogram(val_logits, val_labels, EMOTION_LABELS),
            "val_rank_of_true_class_stats": val_rank_stats,
            "val_rare_class_top1_confusion": val_confusion,
            "val_posthoc_adjustment_sweep": val_sweep,
            "posthoc_candidate_tau_eval": chosen_tau,
            "test_posthoc_adjusted_result": test_posthoc,
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--loss", type=str, choices=["la", "ce"], default="la")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--posthoc_tau",
        type=float,
        default=None,
        help="Frozen post-hoc adjustment tau (skips per-run candidate auto-selection). "
        "Use this to apply a tau value already selected on a prior seed's val split.",
    )
    args = parser.parse_args()

    run_name = args.run_name or f"context_text_seed{args.seed}"
    run_dir = PROJECT_ROOT / "outputs" / run_name

    report = train(args.seed, args.k, args.max_length, args.loss, run_dir, posthoc_tau=args.posthoc_tau)
    print(
        f"[done] run_name={run_name} loss={args.loss} test_weighted_f1={report['test_weighted_f1']:.4f} "
        f"test_macro_f1={report['test_macro_f1']:.4f} test_accuracy={report['test_accuracy']:.4f} "
        f"all_7_classes_nonzero={report['all_7_classes_nonzero']} "
        f"best_epoch={report['best_epoch']} avg_epoch_time_sec={report['avg_epoch_time_sec']:.2f}\n"
        f"[done] posthoc_candidate_tau_eval={report['posthoc_candidate_tau_eval']} "
        f"test_posthoc_adjusted_result={report['test_posthoc_adjusted_result']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
