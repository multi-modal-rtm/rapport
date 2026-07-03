"""Prediction-distribution audit (diagnostics only): loads a trained
speaker_only checkpoint, re-runs test-set inference, and reports the full
predicted-class histogram -- distinguishes "class X is never predicted"
(0 predictions) from "class X is predicted but always wrong" (predictions
exist, just never correct), which a classification_report's precision/recall
alone cannot distinguish when precision is 0 (0/0 and 0/N both round to 0
under zero_division=0).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.data.constants import EMOTION_LABELS
from rapport.models.social_gnn import SocialGNN

NUM_CLASSES = len(EMOTION_LABELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="e.g. outputs/speaker_only_seed42")
    parser.add_argument("--processed-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--cache-dir", default="data/meld/cache", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MELDCachedDataset(args.processed_dir / f"{args.split}.parquet", args.cache_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=4)

    model = SocialGNN(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    checkpoint = torch.load(args.run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[audit_predictions] loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_weighted_f1={checkpoint['val_weighted_f1']:.4f})")

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            logits = model(
                batch["video_feat"], batch["audio_feat"], batch["text_feat"],
                batch["speaker_ids"], batch["dialogue_mask"],
            )
            mask = batch["dialogue_mask"]
            all_preds.extend(logits.argmax(dim=-1)[mask].cpu().tolist())
            all_labels.extend(batch["labels"][mask].cpu().tolist())

    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)

    print(f"\n[audit_predictions] {args.split} split, {len(all_preds)} utterances")
    print(f"{'class':<10} {'true_count':>10} {'predicted_count':>16} {'never_predicted':>16}")
    for i, name in enumerate(EMOTION_LABELS):
        true_c = label_counts.get(i, 0)
        pred_c = pred_counts.get(i, 0)
        print(f"{name:<10} {true_c:>10} {pred_c:>16} {'YES' if pred_c == 0 else 'no':>16}")


if __name__ == "__main__":
    main()
