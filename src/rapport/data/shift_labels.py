"""Emotion-shift auxiliary label derivation (docs/SPEC_RAPPORT_COMPONENTS.md
section B1). Pure functions over a MELD index dataframe -- the actual
parquet mutation lives in scripts/add_shift_labels.py, so this stays
directly unit-testable without file I/O.
"""

from __future__ import annotations

import pandas as pd


def add_shift_labels(df: pd.DataFrame) -> pd.DataFrame:
    """shift_label[t] = 1 iff utterance t's emotion label differs from THE
    SAME SPEAKER's previous utterance label within the dialogue; 0 if same.
    A speaker's FIRST utterance in a dialogue has no shift label --
    shift_mask=0 marks that (shift_label is a dummy 0, never meant to be
    read when shift_mask is 0).
    """
    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)

    shift_label = [0] * len(df)
    shift_mask = [0] * len(df)
    last_label_by_speaker: dict[tuple[int, str], int] = {}

    for idx, row in enumerate(df.itertuples(index=False)):
        key = (row.dialogue_id, row.speaker)
        if key in last_label_by_speaker:
            shift_label[idx] = int(row.label != last_label_by_speaker[key])
            shift_mask[idx] = 1
        last_label_by_speaker[key] = row.label

    df["shift_label"] = shift_label
    df["shift_mask"] = shift_mask
    return df


def compute_shift_pos_weight(shift_label: pd.Series, shift_mask: pd.Series) -> float:
    """pos_weight for BCEWithLogitsLoss (spec B3): num_negative / num_positive
    among MASK-ELIGIBLE rows only (shift_mask == 1) -- upweights the rarer
    class the standard way, matching torch's BCEWithLogitsLoss(pos_weight=...)
    convention (a rarer positive class gets pos_weight > 1).
    """
    eligible = shift_mask == 1
    n_pos = int((shift_label[eligible] == 1).sum())
    n_neg = int((shift_label[eligible] == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive (shift=1) examples among eligible rows -- cannot compute pos_weight")
    return n_neg / n_pos
