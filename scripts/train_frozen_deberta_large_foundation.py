"""Second-encoder replication, Step 2 (docs/PREREG_DeBERTa-v3-large.md):
trains a linear-head text classifier on the FROZEN, non-contextual,
isolated-utterance DeBERTa-v3-large text cache
(data/meld/cache/text_deberta_large/{split}/...), mirroring
scripts/train_frozen_text_foundation.py exactly (encoder swapped, feature
dim 1024 instead of 768) -- the frozen end of this replication's bridging
comparison, under the SAME residual methodology as the fine-tuned end
(RapportModel(residual=True, text_feature_dim=1024)).

Locked optimizer settings match Phase N4's RECIPE.md-locked GNN values,
identical to train_frozen_text_foundation.py: AdamW lr=1e-4,
CosineAnnealingLR(T_max=100), dropout=0.5, early-stop patience 10 on val
MACRO F1, batch=16 dialogues, grad_clip=1.0, bf16 autocast, plain CE loss,
seed 42 (single seed -- mirrors train_frozen_text_foundation.py's own
one-time-anchor convention; the 3-seed variation in this replication's
frozen arm comes from the downstream residual graph stack, not this probe).

Also caches this checkpoint's own 7-d logits for every split to
data/meld/cache/text_deberta_large_logits/{split}/..., so
`--text_cache_subdir text_deberta_large --residual --text_feature_dim 1024`
on scripts/train_rapport.py can consume it directly.

Usage:
    uv run python -m scripts.train_frozen_deberta_large_foundation
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.data.constants import EMOTION_LABELS
from rapport.eval.rank_diagnostics import posthoc_adjustment_sweep
from rapport.eval.report import evaluate_and_report
from rapport.repro import snapshot_environment
from rapport.seed import set_seed
from rapport.training.losses import compute_class_priors

NUM_CLASSES = len(EMOTION_LABELS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LR = 1e-4
DROPOUT = 0.5
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
GRAD_CLIP = 1.0
FROZEN_POSTHOC_TAU = 0.25

SEED = 42
FEATURE_DIM = 1024
TEXT_CACHE_SUBDIR = "text_deberta_large"
LOGITS_OUT_SUBDIR = f"{TEXT_CACHE_SUBDIR}_logits"
RUN_NAME = f"frozen_deberta_large_foundation_seed{SEED}"


class FrozenTextLinearProbe(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = FEATURE_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(text_feat))


def build_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS) -> dict[str, DataLoader]:
    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    cache_dir = PROJECT_ROOT / "data" / "meld" / "cache"
    loaders = {}
    for split in ("train", "dev", "test"):
        dataset = MELDCachedDataset(
            processed_dir / f"{split}.parquet", cache_dir, text_cache_subdir=TEXT_CACHE_SUBDIR, text_feature_dim=FEATURE_DIM
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_dialogues,
            num_workers=num_workers,
        )
    return loaders


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_split(model: FrozenTextLinearProbe, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch["text_feat"])
        mask = batch["dialogue_mask"]
        all_logits.append(logits[mask].cpu())
        all_labels.append(batch["labels"][mask].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=-1).tolist()
    labels_list = labels.tolist()
    weighted_f1 = f1_score(labels_list, preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels_list, preds, average="macro", zero_division=0)
    return weighted_f1, macro_f1, labels_list, preds, logits, labels


def train(seed: int, run_dir: Path) -> tuple[dict, FrozenTextLinearProbe]:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders()
    model = FrozenTextLinearProbe(NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    log_priors = torch.log(compute_class_priors(loaders["train"].dataset.index_df["label"], NUM_CLASSES)).to(device)

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
                logits = model(batch["text_feat"])
                loss = criterion(logits.reshape(-1, NUM_CLASSES), batch["labels"].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        val_weighted_f1, val_macro_f1, val_labels, val_preds, _, _ = evaluate_split(model, loaders["dev"], device)
        per_class_f1 = f1_score(val_labels, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        train_loss = total_loss / max(n_batches, 1)
        per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, per_class_f1)}

        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} val_weighted_f1={val_weighted_f1:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} time={epoch_time:.1f}s per_class_f1={per_class_str}",
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
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_macro_f1": val_macro_f1}, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"[trainer] early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})", flush=True)
                break

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_weighted_f1, test_macro_f1, test_labels, test_preds, test_logits, test_labels_t = evaluate_split(
        model, loaders["test"], device
    )
    test_accuracy = sum(int(p == l) for p, l in zip(test_preds, test_labels)) / len(test_labels)

    report = evaluate_and_report(test_labels, test_preds, EMOTION_LABELS, run_dir)
    per_class_test_f1 = f1_score(test_labels, test_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    posthoc_result = posthoc_adjustment_sweep(
        test_logits, test_labels_t, log_priors.cpu(), [FROZEN_POSTHOC_TAU], EMOTION_LABELS
    )[0]

    report.update(
        {
            "seed": seed,
            "text_cache_subdir": TEXT_CACHE_SUBDIR,
            "feature_dim": FEATURE_DIM,
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_accuracy,
            "test_per_class_f1": {l: float(f1) for l, f1 in zip(EMOTION_LABELS, per_class_test_f1)},
            "test_posthoc_adjusted_result": posthoc_result,
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
    return report, model


def cache_logits(model: FrozenTextLinearProbe, device: torch.device) -> dict[str, int]:
    model.eval()
    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    cache_dir = PROJECT_ROOT / "data" / "meld" / "cache"
    out_dir = cache_dir / LOGITS_OUT_SUBDIR
    counts = {}
    for split in ("train", "dev", "test"):
        dataset = MELDCachedDataset(
            processed_dir / f"{split}.parquet", cache_dir, text_cache_subdir=TEXT_CACHE_SUBDIR, text_feature_dim=FEATURE_DIM
        )
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        with torch.no_grad():
            for idx in range(len(dataset)):
                dialogue = dataset.dialogues[idx]
                text_feat = dataset[idx]["text_feat"].to(device)
                logits = model(text_feat)
                for i, row in enumerate(dialogue.itertuples(index=False)):
                    stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
                    torch.save(logits[i].float().cpu(), split_dir / stem)
                    n_written += 1
        counts[split] = n_written
        print(f"[cache_logits] split={split} wrote {n_written} files", flush=True)

    manifest = {
        "cache_version": "text_deberta_large_frozen_logits_v1",
        "source_checkpoint": f"outputs/{RUN_NAME}/best_model.pt",
        "source_text_cache_subdir": TEXT_CACHE_SUBDIR,
        "feature_dim": FEATURE_DIM,
        "splits": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[cache_logits] wrote {out_dir / 'manifest.json'}", flush=True)
    return counts


def main() -> None:
    run_dir = PROJECT_ROOT / "outputs" / RUN_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {RUN_NAME} already has metrics.json", flush=True)
        report = json.loads(metrics_path.read_text())
        model = FrozenTextLinearProbe(NUM_CLASSES).to(device)
        ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        report, model = train(SEED, run_dir)
        print(
            f"[done] run_name={RUN_NAME} test_weighted_f1={report['test_weighted_f1']:.4f} "
            f"test_macro_f1={report['test_macro_f1']:.4f} test_accuracy={report['test_accuracy']:.4f} "
            f"best_epoch={report['best_epoch']}\n"
            f"[done] test_posthoc_adjusted_result={report['test_posthoc_adjusted_result']}",
            flush=True,
        )

    logits_manifest_path = PROJECT_ROOT / "data" / "meld" / "cache" / LOGITS_OUT_SUBDIR / "manifest.json"
    if logits_manifest_path.exists():
        print(f"[skip] {LOGITS_OUT_SUBDIR} cache already built", flush=True)
    else:
        cache_logits(model, device)


if __name__ == "__main__":
    main()
