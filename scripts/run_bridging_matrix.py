"""BRIDGING EXPERIMENT Step 2: runs full_R and base_fusion_R EXACTLY as in
N5-A (same residual design, same frozen A/V caches) but pointed at the
FROZEN-ERA text foundation (scripts/train_frozen_text_foundation.py,
text_cache_subdir="text") instead of the fine-tuned Phase T contextual
cache (text_ctx). Only the text representation differs across the bridge.

6 runs: {base_fusion_R, full_R} x seeds {42, 1337, 2024}. Idempotent: skips
anything whose metrics.json already exists.

Usage (run as a module):
    uv run python -m scripts.run_bridging_matrix
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.train_rapport import train as train_rapport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1337, 2024)
TEXT_CACHE_SUBDIR = "text"  # frozen-era, non-contextual (cache_version 4)

CONFIGS = {
    "base_fusion_R_frozen": {"relational": False, "shift": False, "temporal": False, "residual": True},
    "full_R_frozen": {"relational": True, "shift": True, "temporal": True, "residual": True},
}


def run(config_name: str, flags: dict, seed: int) -> dict:
    run_name = f"{config_name}_seed{seed}"
    run_dir = PROJECT_ROOT / "outputs" / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {run_name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text())

    print(f"[run] {run_name} (text_cache_subdir={TEXT_CACHE_SUBDIR}, seed={seed})", flush=True)
    start = time.time()
    report = train_rapport(seed=seed, run_dir=run_dir, text_cache_subdir=TEXT_CACHE_SUBDIR, **flags)
    print(f"[run-done] {run_name} wall_clock_sec={time.time() - start:.1f}", flush=True)
    return report


def main() -> None:
    for config_name, flags in CONFIGS.items():
        for seed in SEEDS:
            report = run(config_name, flags, seed)
            print(
                f"[status] {config_name}_seed{seed} test_weighted_f1={report['test_weighted_f1']:.4f} "
                f"all_7_nonzero={report['all_7_classes_nonzero']}",
                flush=True,
            )

    print("[done] bridging matrix complete", flush=True)


if __name__ == "__main__":
    main()
