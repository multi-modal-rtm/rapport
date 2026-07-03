"""Per-modality linear probes on cached frozen-backbone features (diagnostics only).

Utterance-level, no dialogue/context, no GNN — isolates feature quality from
the model. sklearn LogisticRegression trained on train split, evaluated on
test split, reported both with class_weight='balanced' and unbalanced
(default), plus two reference rows (all-neutral constant classifier,
stratified-random classifier) so the modality numbers have context.

Per sign-off: all cross-run comparisons use the UNBALANCED weighted F1
column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
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


def probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, max_iter: int) -> tuple[float, float]:
    unbalanced = LogisticRegression(max_iter=max_iter)
    unbalanced.fit(X_train, y_train)
    wf1_unbalanced = f1_score(y_test, unbalanced.predict(X_test), average="weighted", zero_division=0)

    balanced = LogisticRegression(class_weight="balanced", max_iter=max_iter)
    balanced.fit(X_train, y_train)
    wf1_balanced = f1_score(y_test, balanced.predict(X_test), average="weighted", zero_division=0)

    return wf1_unbalanced, wf1_balanced


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

    rows = []

    # --- reference rows ---
    dummy_neutral = DummyClassifier(strategy="most_frequent")
    dummy_neutral.fit(np.zeros((len(y_train), 1)), y_train)
    wf1_neutral = f1_score(y_test, dummy_neutral.predict(np.zeros((len(y_test), 1))), average="weighted", zero_division=0)
    rows.append(("all-neutral (constant)", wf1_neutral, wf1_neutral))

    dummy_stratified = DummyClassifier(strategy="stratified", random_state=0)
    dummy_stratified.fit(np.zeros((len(y_train), 1)), y_train)
    wf1_stratified = f1_score(
        y_test, dummy_stratified.predict(np.zeros((len(y_test), 1))), average="weighted", zero_division=0
    )
    rows.append(("random-stratified", wf1_stratified, wf1_stratified))

    # --- per-modality probes ---
    for m in MODALITIES:
        wf1_unbalanced, wf1_balanced = probe(train_feats[m], y_train, test_feats[m], y_test, args.max_iter)
        rows.append((m, wf1_unbalanced, wf1_balanced))
        print(f"[probe] {m}: unbalanced={wf1_unbalanced:.4f} balanced={wf1_balanced:.4f}")

    X_train_concat = np.concatenate([train_feats[m] for m in MODALITIES], axis=1)
    X_test_concat = np.concatenate([test_feats[m] for m in MODALITIES], axis=1)
    wf1_unbalanced, wf1_balanced = probe(X_train_concat, y_train, X_test_concat, y_test, args.max_iter)
    rows.append(("concat", wf1_unbalanced, wf1_balanced))
    print(f"[probe] concat: unbalanced={wf1_unbalanced:.4f} balanced={wf1_balanced:.4f}")

    print("\n[probe] === summary (weighted F1) ===")
    print(f"{'row':<24} {'unbalanced':>12} {'balanced':>12}")
    for name, wf1_u, wf1_b in rows:
        print(f"{name:<24} {wf1_u:>12.4f} {wf1_b:>12.4f}")
    print("\n[probe] per sign-off: use the UNBALANCED column for all cross-run comparisons.")


if __name__ == "__main__":
    main()
