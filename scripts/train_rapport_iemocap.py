"""Phase N5-B Step B5.1: trains RapportModel (base_fusion_R / full_R /
minus_relational_R configs) over the frozen IEMOCAP text_ctx_iemocap + A/V
caches. Identical apparatus to MELD's scripts/train_rapport.py (that
script itself untouched -- this is a parallel copy, same pattern as every
other IEMOCAP mirror script in this phase), repointed at
data/iemocap/processed / data/iemocap/cache and the 6-class label set.

Scoring throughout Phase B3 is RAW (unadjusted) logits -- B4 found no
post-hoc tau candidate qualified under the locked selection rule, so this
script does not compute or report a post-hoc-adjusted secondary score at
all (MELD's train_rapport.py's FROZEN_POSTHOC_TAU=0.25 has no IEMOCAP
analog; carrying it over would silently reintroduce an adjustment this
project's own B4 result says isn't warranted here).

Usage:
    uv run python scripts/train_rapport_iemocap.py --seed 42 --run_name base_fusion_R_iemocap_seed42 --residual
    uv run python scripts/train_rapport_iemocap.py --seed 42 --run_name full_R_iemocap_seed42 --residual --relational --shift --temporal
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

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.data.constants import IEMOCAP_EMOTION_LABELS as EMOTION_LABELS
from rapport.data.shift_labels import compute_shift_pos_weight
from rapport.eval.report import evaluate_and_report
from rapport.models.rapport_model import RapportModel
from rapport.repro import snapshot_environment
from rapport.seed import set_seed
from rapport.training.losses import compute_class_priors

NUM_CLASSES = len(EMOTION_LABELS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "iemocap" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "iemocap" / "cache"

# RECIPE.md-locked GNN values -- identical apparatus to MELD, unchanged.
LR = 1e-4
DROPOUT = 0.5
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
GRAD_CLIP = 1.0

SHIFT_LOSS_WEIGHT = 0.5


def build_dataloaders(
    temporal: bool = False,
    residual: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    text_cache_subdir: str = "text_ctx_iemocap",
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MELDCachedDataset(
            PROCESSED_DIR / f"{split}.parquet",
            CACHE_DIR,
            text_cache_subdir=text_cache_subdir,
            load_av_tokens=temporal,
            load_text_logits=residual,
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


def _model_forward(model: RapportModel, batch: dict):
    kwargs = {}
    if model.temporal:
        kwargs.update(
            video_tokens=batch["video_tokens"], video_tokens_mask=batch["video_tokens_mask"],
            audio_tokens=batch["audio_tokens"], audio_tokens_mask=batch["audio_tokens_mask"],
        )
    if model.residual:
        kwargs["text_logits"] = batch["text_logits"]
    return model(
        batch["video_feat"], batch["audio_feat"], batch["text_feat"], batch["speaker_ids"], batch["dialogue_mask"],
        **kwargs,
    )


@torch.no_grad()
def evaluate_split(model: RapportModel, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    all_shift_logits, all_shift_labels = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits, shift_logits = _model_forward(model, batch)
        mask = batch["dialogue_mask"]
        all_logits.append(logits[mask].cpu())
        all_labels.append(batch["labels"][mask].cpu())

        if model.shift:
            shift_valid = mask & (batch["shift_mask"] > 0.5)
            all_shift_logits.append(shift_logits[shift_valid].cpu())
            all_shift_labels.append(batch["shift_label"][shift_valid].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=-1).tolist()
    labels_list = labels.tolist()
    weighted_f1 = f1_score(labels_list, preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(labels_list, preds, average="macro", zero_division=0)

    shift_f1 = None
    if model.shift:
        shift_logits_all = torch.cat(all_shift_logits)
        shift_labels_all = torch.cat(all_shift_labels)
        shift_preds = (shift_logits_all > 0).long().tolist()
        shift_f1 = f1_score(shift_labels_all.long().tolist(), shift_preds, average="binary", zero_division=0)

    return weighted_f1, macro_f1, labels_list, preds, logits, labels, shift_f1


def train(
    seed: int,
    relational: bool,
    shift: bool,
    temporal: bool,
    run_dir: Path,
    residual: bool = False,
    text_cache_subdir: str = "text_ctx_iemocap",
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(temporal=temporal, residual=residual, text_cache_subdir=text_cache_subdir)
    model = RapportModel(
        num_classes=NUM_CLASSES,
        relational=relational,
        shift=shift,
        temporal=temporal,
        residual=residual,
        dropout=DROPOUT,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    shift_criterion = None
    if shift:
        train_df = loaders["train"].dataset.index_df
        pos_weight = compute_shift_pos_weight(train_df["shift_label"], train_df["shift_mask"])
        shift_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        print(f"[trainer] shift pos_weight={pos_weight:.4f}", flush=True)

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
                logits, shift_logits = _model_forward(model, batch)
                loss = criterion(logits.reshape(-1, NUM_CLASSES), batch["labels"].reshape(-1))
                if shift:
                    shift_valid = batch["dialogue_mask"] & (batch["shift_mask"] > 0.5)
                    if shift_valid.any():
                        shift_loss = shift_criterion(shift_logits[shift_valid], batch["shift_label"][shift_valid])
                        loss = loss + SHIFT_LOSS_WEIGHT * shift_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        val_weighted_f1, val_macro_f1, val_labels, val_preds, _, _, val_shift_f1 = evaluate_split(
            model, loaders["val"], device
        )
        per_class_f1 = f1_score(val_labels, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        train_loss = total_loss / max(n_batches, 1)
        per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, per_class_f1)}
        shift_f1_str = f" val_shift_f1={val_shift_f1:.4f}" if shift else ""

        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} val_weighted_f1={val_weighted_f1:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}{shift_f1_str} time={epoch_time:.1f}s per_class_f1={per_class_str}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_weighted_f1": val_weighted_f1,
                "val_macro_f1": val_macro_f1,
                "val_shift_f1": val_shift_f1,
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
    test_weighted_f1, test_macro_f1, test_labels, test_preds, test_logits, test_labels_t, test_shift_f1 = evaluate_split(
        model, loaders["test"], device
    )
    test_accuracy = sum(int(p == l) for p, l in zip(test_preds, test_labels)) / len(test_labels)

    report = evaluate_and_report(test_labels, test_preds, EMOTION_LABELS, run_dir)
    per_class_test_f1 = f1_score(test_labels, test_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    report.update(
        {
            "seed": seed,
            "relational": relational,
            "shift": shift,
            "temporal": temporal,
            "residual": residual,
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
            "test_shift_f1": test_shift_f1,
            "test_accuracy": test_accuracy,
            "test_per_class_f1": {l: float(f1) for l, f1 in zip(EMOTION_LABELS, per_class_test_f1)},
            "test_posthoc_adjusted_result": None,  # B4: no qualifying tau on this encoder; scored RAW throughout.
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": checkpoint["epoch"],
            "num_epochs_run": len(epoch_times),
            "epoch_times_sec": epoch_times,
            "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
            "history": history,
            "all_6_classes_nonzero": bool(all(f1 > 0 for f1 in per_class_test_f1)),
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--relational", action="store_true")
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--text_cache_subdir", type=str, default="text_ctx_iemocap")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / args.run_name
    report = train(
        args.seed, args.relational, args.shift, args.temporal, run_dir,
        residual=args.residual, text_cache_subdir=args.text_cache_subdir,
    )
    print(
        f"[done] run_name={args.run_name} test_weighted_f1={report['test_weighted_f1']:.4f} "
        f"test_macro_f1={report['test_macro_f1']:.4f} test_accuracy={report['test_accuracy']:.4f} "
        f"all_6_classes_nonzero={report['all_6_classes_nonzero']} best_epoch={report['best_epoch']} "
        f"avg_epoch_time_sec={report['avg_epoch_time_sec']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
