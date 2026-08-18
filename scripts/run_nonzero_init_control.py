"""REVIEWER-RESPONSE (Major Concern #2 / Q2): does exact-zero-init of
W_out and the A/V fusion blocks under-credit the graph stack on a
fine-tuned foundation via early-epoch gradient starvation / a degenerate
local minimum, rather than the components genuinely having no value?

Empirical test: reruns the Full stack (relational=shift=temporal=
residual=True) on the SAME fine-tuned MELD foundation as full_R_seed{42,
1337,2024} (text_cache_subdir="text_ctx", the locked k=8 recipe), 3
identical seeds, with residual_init_scale=NONZERO_SCALE instead of the
spec v1.1 default (exact zero) -- see rapport_model.py's
residual_init_scale parameter, default-0.0-preserving addition. Every
other hyperparameter (LR, dropout, epochs, patience, batch size, grad
clip, shift loss weight) is untouched.

Run names: full_R_nonzeroinit_seed{seed} (distinct from the existing
full_R_seed{seed} zero-init runs, which this script never overwrites).

Usage:
    uv run python -m scripts.run_nonzero_init_control
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.train_rapport import train as train_rapport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1337, 2024)
NONZERO_SCALE = 0.02  # small: symmetry-breaking, not a strong departure from near-identity init
FULL_R_FLAGS = {"relational": True, "shift": True, "temporal": True, "residual": True}


def run_name(seed: int) -> str:
    return f"full_R_nonzeroinit_seed{seed}"


def run_one(seed: int) -> dict:
    name = run_name(seed)
    run_dir = PROJECT_ROOT / "outputs" / name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text())

    print(f"[run] {name} (residual_init_scale={NONZERO_SCALE}, seed={seed})", flush=True)
    start = time.time()
    report = train_rapport(
        seed=seed, run_dir=run_dir, text_cache_subdir="text_ctx",
        residual_init_scale=NONZERO_SCALE, **FULL_R_FLAGS,
    )
    print(f"[run-done] {name} wall_clock_sec={time.time() - start:.1f}", flush=True)
    return report


def main() -> None:
    for seed in SEEDS:
        report = run_one(seed)
        print(
            f"[status] {run_name(seed)} test_weighted_f1={report['test_weighted_f1']:.4f} "
            f"best_epoch={report['best_epoch']}",
            flush=True,
        )
    print("[done] non-zero-init control complete", flush=True)


if __name__ == "__main__":
    main()
