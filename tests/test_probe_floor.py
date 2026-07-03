"""Regression guard for the RAPPORT hard floor (docs/PIVOT.md): any full
multimodal model must beat this same text-only probe's unbalanced weighted
F1 by >= 0.03. If the cached text features silently regress (wrong pooling,
stale/partial rebuild, wrong cache_version), every downstream gate shifts
with it -- catch that here directly instead of only noticing it inside a
multi-hour training run's gate check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "meld" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "meld" / "cache"

PROBE_FLOOR = 0.50


def _load_text_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
    feats, labels = [], []
    for row in df.itertuples(index=False):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        feats.append(torch.load(CACHE_DIR / "text" / split / stem, weights_only=True).numpy())
        labels.append(row.label)
    return np.stack(feats), np.array(labels)


@pytest.mark.slow
def test_cached_text_probe_meets_floor():
    X_train, y_train = _load_text_features("train")
    X_test, y_test = _load_text_features("test")

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    weighted_f1 = f1_score(y_test, clf.predict(X_test), average="weighted", zero_division=0)

    assert weighted_f1 >= PROBE_FLOOR, (
        f"cached text probe unbalanced weighted F1 {weighted_f1:.4f} fell below the "
        f"{PROBE_FLOOR} floor -- the text cache may have regressed (wrong pooling, "
        "stale/partial rebuild, wrong cache_version)"
    )
