"""Per-modality linear probes on cached frozen-backbone features (diagnostics only).

Utterance-level, no dialogue/context, no GNN — isolates feature quality from
the model. sklearn LogisticRegression(class_weight='balanced') trained on
train split, evaluated on test split. One probe per modality plus one on the
concatenated 3x768 vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

MODALITIES = ("video", "audio", "text")


def load_features(processed_dir: Path, cache_dir: Path, split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    df = pd.read_parquet(processed_dir / f"{split}.parquet")
    feats = {m: [] for m in MODALITIES}
    labels = []
    for row in df.itertuples(index=False):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        for m in MODALITIES:
            feats[m].append(torch.load(cache_dir / m / split / stem, weights_only=True).numpy())
        labels.append(row.label)
    return {m: np.stack(v) for m, v in feats.items()}, np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--cache-dir", default="data/meld/cache", type=Path)
    parser.add_argument("--max-iter", default=2000, type=int)
    args = parser.parse_args()

    print("[probe] loading train features...")
    train_feats, y_train = load_features(args.processed_dir, args.cache_dir, "train")
    print("[probe] loading test features...")
    test_feats, y_test = load_features(args.processed_dir, args.cache_dir, "test")

    results = {}
    for m in MODALITIES:
        clf = LogisticRegression(class_weight="balanced", max_iter=args.max_iter)
        clf.fit(train_feats[m], y_train)
        preds = clf.predict(test_feats[m])
        wf1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        results[m] = wf1
        print(f"[probe] {m}: weighted F1 = {wf1:.4f}")

    X_train_concat = np.concatenate([train_feats[m] for m in MODALITIES], axis=1)
    X_test_concat = np.concatenate([test_feats[m] for m in MODALITIES], axis=1)
    clf = LogisticRegression(class_weight="balanced", max_iter=args.max_iter)
    clf.fit(X_train_concat, y_train)
    preds = clf.predict(X_test_concat)
    wf1_concat = f1_score(y_test, preds, average="weighted", zero_division=0)
    results["concat"] = wf1_concat
    print(f"[probe] concat: weighted F1 = {wf1_concat:.4f}")

    print("\n[probe] summary:", {k: round(v, 4) for k, v in results.items()})


if __name__ == "__main__":
    main()
