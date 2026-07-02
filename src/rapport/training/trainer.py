"""Training loop for the v1 SocialArcNet baseline (Social GNN over cached,
frozen-backbone V/A/T features — configs A and B).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.data.constants import EMOTION_LABELS
from rapport.eval.report import evaluate_and_report
from rapport.models.social_gnn import SocialGNN
from rapport.training.losses import FocalLoss

NUM_CLASSES = len(EMOTION_LABELS)


def build_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    processed_dir = Path(cfg.paths.data_dir) / "meld" / "processed"
    cache_dir = Path(cfg.paths.data_dir) / "meld" / "cache"
    loaders = {}
    for split in ("train", "dev", "test"):
        dataset = MELDCachedDataset(processed_dir / f"{split}.parquet", cache_dir)
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_dialogues,
            num_workers=cfg.training.num_workers,
        )
    return loaders


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_split(model: SocialGNN, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(
            batch["video_feat"], batch["audio_feat"], batch["text_feat"],
            batch["speaker_ids"], batch["dialogue_mask"],
        )
        mask = batch["dialogue_mask"]
        all_preds.extend(logits.argmax(dim=-1)[mask].cpu().tolist())
        all_labels.extend(batch["labels"][mask].cpu().tolist())

    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return weighted_f1, macro_f1, all_labels, all_preds


def train(cfg: DictConfig, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(cfg)
    model = SocialGNN(num_classes=NUM_CLASSES, dropout=cfg.training.dropout).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)
    criterion = FocalLoss(gamma=cfg.training.focal_gamma, ignore_index=-1)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    ckpt_path = run_dir / "best_model.pt"

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    epoch_times: list[float] = []
    history: list[dict] = []

    for epoch in range(cfg.training.max_epochs):
        start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loaders["train"]:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(
                    batch["video_feat"], batch["audio_feat"], batch["text_feat"],
                    batch["speaker_ids"], batch["dialogue_mask"],
                )
                loss = criterion(logits.reshape(-1, NUM_CLASSES), batch["labels"].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        val_weighted_f1, val_macro_f1, val_labels, val_preds = evaluate_split(model, loaders["dev"], device)
        per_class_f1 = f1_score(
            val_labels, val_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
        )
        train_loss = total_loss / max(n_batches, 1)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/weighted_f1", val_weighted_f1, epoch)
        writer.add_scalar("val/macro_f1", val_macro_f1, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        for label, f1 in zip(EMOTION_LABELS, per_class_f1):
            writer.add_scalar(f"val/f1_{label}", f1, epoch)

        per_class_str = {label: round(float(f1), 3) for label, f1 in zip(EMOTION_LABELS, per_class_f1)}
        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} val_weighted_f1={val_weighted_f1:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} time={epoch_time:.1f}s per_class_f1={per_class_str}"
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

        if val_weighted_f1 > best_val_f1:
            best_val_f1 = val_weighted_f1
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_weighted_f1": val_weighted_f1},
                ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.early_stop_patience:
                print(f"[trainer] early stopping at epoch {epoch} (patience={cfg.training.early_stop_patience})")
                break

    writer.close()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_weighted_f1, test_macro_f1, test_labels, test_preds = evaluate_split(model, loaders["test"], device)
    test_accuracy = sum(int(p == l) for p, l in zip(test_preds, test_labels)) / len(test_labels)

    report = evaluate_and_report(test_labels, test_preds, EMOTION_LABELS, run_dir)
    report.update(
        {
            "test_weighted_f1": test_weighted_f1,
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_accuracy,
            "best_val_weighted_f1": best_val_f1,
            "best_epoch": checkpoint["epoch"],
            "num_epochs_run": len(epoch_times),
            "epoch_times_sec": epoch_times,
            "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
            "history": history,
        }
    )
    return report
