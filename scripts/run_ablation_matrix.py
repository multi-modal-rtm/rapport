"""Phase N4 Step 5: runs the full ablation matrix --
{full, minus_relational, minus_shift, minus_temporal, base_fusion} x seeds
{42, 1337, 2024} = 15 runs -- via scripts.train_rapport.train.

Idempotent: skips any (config, seed) whose outputs/<run_name>/metrics.json
already exists, so an interrupted or partially-completed matrix can be
safely resumed by just rerunning this script. `base_fusion_seed42` from
Phase N4 Step 1 is reused this way rather than retrained.

Usage (run as a module -- it imports scripts.train_rapport):
    uv run python -m scripts.run_ablation_matrix
    uv run python -m scripts.run_ablation_matrix --configs full base_fusion --seeds 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.train_rapport import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# name -> (relational, shift, temporal)
CONFIGS: dict[str, dict[str, bool]] = {
    "full": {"relational": True, "shift": True, "temporal": True},
    "minus_relational": {"relational": False, "shift": True, "temporal": True},
    "minus_shift": {"relational": True, "shift": False, "temporal": True},
    "minus_temporal": {"relational": True, "shift": True, "temporal": False},
    "base_fusion": {"relational": False, "shift": False, "temporal": False},
}
SEEDS = (42, 1337, 2024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    results = {}
    for config_name in args.configs:
        flags = CONFIGS[config_name]
        for seed in args.seeds:
            run_name = f"{config_name}_seed{seed}"
            run_dir = PROJECT_ROOT / "outputs" / run_name
            metrics_path = run_dir / "metrics.json"

            if metrics_path.exists():
                print(f"[skip] {run_name} already has metrics.json", flush=True)
                report = json.loads(metrics_path.read_text())
            else:
                print(f"[run] {run_name} flags={flags}", flush=True)
                start = time.time()
                report = train(seed=seed, run_dir=run_dir, **flags)
                elapsed = time.time() - start
                print(f"[run-done] {run_name} wall_clock_sec={elapsed:.1f}", flush=True)

            results[run_name] = {
                "test_weighted_f1": report["test_weighted_f1"],
                "test_macro_f1": report["test_macro_f1"],
                "test_posthoc_adjusted_result": report["test_posthoc_adjusted_result"],
                "all_7_classes_nonzero": report["all_7_classes_nonzero"],
            }
            print(
                f"[status] {run_name} test_weighted_f1={report['test_weighted_f1']:.4f} "
                f"test_macro_f1={report['test_macro_f1']:.4f} all_7_nonzero={report['all_7_classes_nonzero']}",
                flush=True,
            )

    summary_path = PROJECT_ROOT / "outputs" / "ablation_matrix_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
