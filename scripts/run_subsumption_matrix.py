"""Phase N5-A: the subsumption curve. Trains base_fusion_R and full_R
(residual=True) on each of the k in {0, 2, 4, 8} text_ctx caches built by
scripts/train_context_text.py + scripts/build_text_ctx_cache.py.

k=8 reuses Phase N4-R Step 4's existing runs (base_fusion_R_seed{42,1337,2024},
full_R_seed{42,1337,2024}) and the original "text_ctx"/"text_ctx_logits"
cache -- no k-suffix in those run names. k=0 and k=8 get all 3 seeds
(error bars, per the phase spec); k=2/4 are seed 42 only.

Idempotent: skips any (config, k, seed) whose outputs/<run_name>/metrics.json
already exists.

Usage (run as a module):
    uv run python -m scripts.run_subsumption_matrix
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.train_rapport import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS: dict[str, dict[str, bool]] = {
    "base_fusion_R": {"relational": False, "shift": False, "temporal": False, "residual": True},
    "full_R": {"relational": True, "shift": True, "temporal": True, "residual": True},
}
K_VALUES = (0, 2, 4, 8)
ENDPOINT_SEEDS = (42, 1337, 2024)
MIDPOINT_SEEDS = (42,)


def text_cache_subdir_for_k(k: int) -> str:
    return "text_ctx" if k == 8 else f"text_ctx_k{k}"


def run_name_for(config_name: str, k: int, seed: int) -> str:
    if k == 8:
        return f"{config_name}_seed{seed}"  # reuse Phase N4-R Step 4 naming
    return f"{config_name}_k{k}_seed{seed}"


def main() -> None:
    results = {}
    for k in K_VALUES:
        seeds = ENDPOINT_SEEDS if k in (0, 8) else MIDPOINT_SEEDS
        text_cache_subdir = text_cache_subdir_for_k(k)
        for config_name, flags in CONFIGS.items():
            for seed in seeds:
                run_name = run_name_for(config_name, k, seed)
                run_dir = PROJECT_ROOT / "outputs" / run_name
                metrics_path = run_dir / "metrics.json"

                if metrics_path.exists():
                    print(f"[skip] {run_name} already has metrics.json", flush=True)
                    report = json.loads(metrics_path.read_text())
                else:
                    print(f"[run] {run_name} k={k} flags={flags} text_cache_subdir={text_cache_subdir}", flush=True)
                    start = time.time()
                    report = train(seed=seed, run_dir=run_dir, text_cache_subdir=text_cache_subdir, **flags)
                    elapsed = time.time() - start
                    print(f"[run-done] {run_name} wall_clock_sec={elapsed:.1f}", flush=True)

                key = f"{config_name}_k{k}_seed{seed}"
                results[key] = {
                    "config": config_name,
                    "k": k,
                    "seed": seed,
                    "test_weighted_f1": report["test_weighted_f1"],
                    "test_macro_f1": report["test_macro_f1"],
                    "all_7_classes_nonzero": report["all_7_classes_nonzero"],
                }
                print(
                    f"[status] {run_name} test_weighted_f1={report['test_weighted_f1']:.4f} "
                    f"test_macro_f1={report['test_macro_f1']:.4f} all_7_nonzero={report['all_7_classes_nonzero']}",
                    flush=True,
                )

    summary_path = PROJECT_ROOT / "outputs" / "subsumption_matrix_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
