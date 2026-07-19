"""Phase N5-B Step B3.1: the pre-registered residual matrix on IEMOCAP,
identical apparatus to MELD (docs/PHASE_N4R.md spec v1.1, docs/PHASE_N5A.md):
{base_fusion_R, full_R, minus_relational_R} x seeds {42, 1337, 2024} = 9
runs. Idempotent: skips anything whose metrics.json already exists.

Writes outputs/iemocap_residual_matrix_DONE.sentinel on completion, so a
watcher process can poll for it (or for this process's own PID) instead of
tailing the log unboundedly (docs/RECIPE.md's monitoring-fix standard,
Phase N5-B Step B3).

Usage (run as a module):
    uv run python -m scripts.run_iemocap_residual_matrix
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.train_rapport_iemocap import train as train_rapport_iemocap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1337, 2024)
DONE_SENTINEL = PROJECT_ROOT / "outputs" / "iemocap_residual_matrix_DONE.sentinel"

CONFIGS = {
    "base_fusion_R_iemocap": {"relational": False, "shift": False, "temporal": False, "residual": True},
    "full_R_iemocap": {"relational": True, "shift": True, "temporal": True, "residual": True},
    "minus_relational_R_iemocap": {"relational": False, "shift": True, "temporal": True, "residual": True},
}


def run(config_name: str, flags: dict, seed: int) -> dict:
    run_name = f"{config_name}_seed{seed}"
    run_dir = PROJECT_ROOT / "outputs" / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {run_name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text())

    print(f"[run] {run_name} (seed={seed})", flush=True)
    start = time.time()
    report = train_rapport_iemocap(seed=seed, run_dir=run_dir, **flags)
    print(f"[run-done] {run_name} wall_clock_sec={time.time() - start:.1f}", flush=True)
    return report


def main() -> None:
    DONE_SENTINEL.unlink(missing_ok=True)
    for config_name, flags in CONFIGS.items():
        for seed in SEEDS:
            report = run(config_name, flags, seed)
            print(
                f"[status] {config_name}_seed{seed} test_weighted_f1={report['test_weighted_f1']:.4f} "
                f"all_6_nonzero={report['all_6_classes_nonzero']}",
                flush=True,
            )

    DONE_SENTINEL.write_text("done\n")
    print("[done] IEMOCAP residual matrix complete", flush=True)


if __name__ == "__main__":
    main()
