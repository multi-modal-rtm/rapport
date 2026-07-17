"""Phase N5-A consolidation Step 2: powers the k=0/k=8 endpoints up to n=7
by adding 4 seeds {7, 123, 555, 9090} to both the text-anchor curve (a
genuine text-encoder retrain at each k) and the "graph config" used in the
figure (full_R, trained on the EXISTING k=0/k=8 caches -- same convention
as the original 3 seeds, which also reused one frozen encoder per k).

16 runs: 4 seeds x 2 k-values x 2 series (8 text-encoder retrains + 8
full_R downstream runs). Idempotent: skips anything whose metrics.json
already exists.

Usage (run as a module):
    uv run python -m scripts.run_n7_powerup
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import scripts.train_context_text as tct
from scripts.train_rapport import train as train_rapport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_SEEDS = (7, 123, 555, 9090)
K_ENDPOINTS = (0, 8)

FULL_R_FLAGS = {"relational": True, "shift": True, "temporal": True, "residual": True}


def text_anchor_run_name(k: int, seed: int) -> str:
    if k == 8:
        return f"context_text_plain_ce_seed{seed}"
    return f"context_text_k{k}_seed{seed}"


def rapport_run_name(k: int, seed: int) -> str:
    if k == 8:
        return f"full_R_seed{seed}"
    return f"full_R_k{k}_seed{seed}"


def text_cache_subdir_for_k(k: int) -> str:
    return "text_ctx" if k == 8 else f"text_ctx_k{k}"


def run_text_anchor(k: int, seed: int) -> dict:
    run_name = text_anchor_run_name(k, seed)
    run_dir = PROJECT_ROOT / "outputs" / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {run_name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text())

    print(f"[run] {run_name} (text encoder, k={k}, seed={seed})", flush=True)
    start = time.time()
    report = tct.train(seed=seed, k=k, max_length=tct.DEFAULT_MAX_LENGTH, loss_kind="ce", run_dir=run_dir)
    print(f"[run-done] {run_name} wall_clock_sec={time.time() - start:.1f}", flush=True)
    return report


def run_full_r(k: int, seed: int) -> dict:
    run_name = rapport_run_name(k, seed)
    run_dir = PROJECT_ROOT / "outputs" / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {run_name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text())

    print(f"[run] {run_name} (full_R, k={k}, seed={seed})", flush=True)
    start = time.time()
    report = train_rapport(seed=seed, run_dir=run_dir, text_cache_subdir=text_cache_subdir_for_k(k), **FULL_R_FLAGS)
    print(f"[run-done] {run_name} wall_clock_sec={time.time() - start:.1f}", flush=True)
    return report


def main() -> None:
    # Text-encoder retrains first (cheap, ~100s each) -- doesn't affect the
    # existing text_ctx_k0/text_ctx caches full_R reads from (those stay
    # pinned to the original seed-42 encoders throughout, matching the
    # original 3-seed convention).
    for k in K_ENDPOINTS:
        for seed in NEW_SEEDS:
            report = run_text_anchor(k, seed)
            print(
                f"[status] {text_anchor_run_name(k, seed)} test_weighted_f1={report['test_weighted_f1']:.4f}",
                flush=True,
            )

    for k in K_ENDPOINTS:
        for seed in NEW_SEEDS:
            report = run_full_r(k, seed)
            print(
                f"[status] {rapport_run_name(k, seed)} test_weighted_f1={report['test_weighted_f1']:.4f} "
                f"all_7_nonzero={report['all_7_classes_nonzero']}",
                flush=True,
            )

    print("[done] n=7 power-up complete", flush=True)


if __name__ == "__main__":
    main()
