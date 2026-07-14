"""Phase N4 Step 3, spec B1: derives emotion-shift labels into the
processed MELD index parquets (data/meld/processed/{split}.parquet), in
place -- never computed on the fly during training. Logic lives in
rapport.data.shift_labels (unit-tested there); this script is the
file-I/O driver.

Idempotent: safe to rerun any time (recomputes and overwrites the two
columns rather than erroring if they already exist), including after a
full scripts/preprocess_meld.py rebuild.

Usage:
    uv run python scripts/add_shift_labels.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rapport.data.shift_labels import add_shift_labels, compute_shift_pos_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "meld" / "processed"
SPLITS = ("train", "dev", "test")


def main() -> None:
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        df = pd.read_parquet(path)
        df = add_shift_labels(df)
        df.to_parquet(path, index=False)

        n_eligible = int(df["shift_mask"].sum())
        n_shift = int((df["shift_label"] * df["shift_mask"]).sum())
        pos_weight = compute_shift_pos_weight(df["shift_label"], df["shift_mask"])
        print(
            f"[add_shift_labels] split={split} rows={len(df)} eligible={n_eligible} "
            f"({n_eligible / len(df):.1%}) shift_rate_among_eligible={n_shift / max(n_eligible, 1):.4f} "
            f"pos_weight={pos_weight:.4f}"
        )


if __name__ == "__main__":
    main()
