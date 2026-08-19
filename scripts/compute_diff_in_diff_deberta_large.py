"""Second-encoder replication, Steps 4/5 (docs/PREREG_DeBERTa-v3-large.md):
computes the frozen-vs-fine-tuned difference-in-differences for the
DeBERTa-v3-large replication, using the SAME bootstrap method as
scripts/compute_diff_in_diff.py (imported directly, not reimplemented):
independent-groups percentile bootstrap, 100,000 resamples, fixed RNG seed
20260818. Reads only this replication's own metrics.json files (all listed
under RUN NAMES below) -- no existing stored value from any prior
experiment is read or changed. Writes docs/DeBERTa-v3-large_REPLICATION.md.

Pairing convention (matches this project's established convention, verified
against docs/DIFF_IN_DIFF.md's raw inputs): frozen paired diffs are each
frozen-condition full-stack seed's test weighted F1 minus the SINGLE frozen
linear-probe anchor's F1 (one-time anchor, reused for all 3 graph-stack
seeds, mirroring scripts/train_frozen_text_foundation.py's own convention).
Fine-tuned paired diffs are each fine-tuned-condition full-stack seed's test
weighted F1 minus the SAME-SEED-NUMBERED text anchor's own F1 (even though
the residual stack's cached z_text/text_ctx features are frozen from the
seed-42 foundation only, per docs/RECIPE.md's "Phase N4 onward" convention)
-- paired by seed number, not by which checkpoint produced the cache.

Usage:
    uv run python -m scripts.compute_diff_in_diff_deberta_large
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.compute_diff_in_diff import N_BOOTSTRAP, RNG_SEED, bootstrap_did

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

SEEDS = (42, 1337, 2024)
FROZEN_ANCHOR_RUN = "frozen_deberta_large_foundation_seed42"
FINETUNED_ANCHOR_RUN_TMPL = "deberta_large_text_anchor_seed{seed}"
FROZEN_FULL_R_RUN_TMPL = "full_R_deberta_large_frozen_seed{seed}"
FINETUNED_FULL_R_RUN_TMPL = "full_R_deberta_large_finetuned_seed{seed}"


def _load_metrics(run_name: str) -> dict:
    path = OUTPUTS_DIR / run_name / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing metrics.json for {run_name} -- run the replication pipeline first")
    return json.loads(path.read_text())


def main() -> None:
    finetuned_anchor = {seed: _load_metrics(FINETUNED_ANCHOR_RUN_TMPL.format(seed=seed)) for seed in SEEDS}
    frozen_anchor = _load_metrics(FROZEN_ANCHOR_RUN)
    frozen_full_r = {seed: _load_metrics(FROZEN_FULL_R_RUN_TMPL.format(seed=seed)) for seed in SEEDS}
    finetuned_full_r = {seed: _load_metrics(FINETUNED_FULL_R_RUN_TMPL.format(seed=seed)) for seed in SEEDS}

    frozen_anchor_f1 = frozen_anchor["test_weighted_f1"]
    finetuned_anchor_f1 = {seed: finetuned_anchor[seed]["test_weighted_f1"] for seed in SEEDS}
    frozen_gain = {seed: frozen_full_r[seed]["test_weighted_f1"] - frozen_anchor_f1 for seed in SEEDS}
    finetuned_gain = {seed: finetuned_full_r[seed]["test_weighted_f1"] - finetuned_anchor_f1[seed] for seed in SEEDS}

    frozen_diffs = list(frozen_gain.values())
    finetuned_diffs = list(finetuned_gain.values())

    rng = np.random.default_rng(RNG_SEED)
    point = float(np.mean(frozen_diffs) - np.mean(finetuned_diffs))
    boot = bootstrap_did(frozen_diffs, finetuned_diffs, N_BOOTSTRAP, rng)
    ci_lo, ci_hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    excludes_zero = bool(ci_lo > 0 or ci_hi < 0)
    frac_frozen_exceeds = float((boot > 0).mean())

    # Reading rule is stated directly in terms of the fine-tuned-regime paired
    # gain itself (not the DiD), so evaluate it on finetuned_diffs directly.
    finetuned_gain_mean = float(np.mean(finetuned_diffs))
    finetuned_gain_all_positive = all(d > 0 for d in finetuned_diffs)
    # A simple percentile bootstrap CI on the fine-tuned-regime gain alone
    # (n=3), for the reading rule's "positive and interval excludes zero" test.
    rng2 = np.random.default_rng(RNG_SEED)
    boot_finetuned_only = rng2.choice(np.asarray(finetuned_diffs), size=(N_BOOTSTRAP, len(finetuned_diffs)), replace=True).mean(axis=1)
    ft_ci_lo, ft_ci_hi = (float(x) for x in np.percentile(boot_finetuned_only, [2.5, 97.5]))
    ft_excludes_zero_positive = bool(ft_ci_lo > 0)

    replicates = not (finetuned_gain_mean > 0 and ft_excludes_zero_positive)
    verdict = "REPLICATES" if replicates else "DOES NOT REPLICATE"

    print(
        f"[compute_diff_in_diff_deberta_large] frozen_anchor_f1={frozen_anchor_f1:.4f} "
        f"finetuned_anchor_f1={finetuned_anchor_f1} frozen_gain={frozen_gain} finetuned_gain={finetuned_gain}",
        flush=True,
    )
    print(
        f"[compute_diff_in_diff_deberta_large] DiD={point:+.4f} 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
        f"excludes_zero={excludes_zero} verdict={verdict}",
        flush=True,
    )

    result = {
        "frozen_anchor_run": FROZEN_ANCHOR_RUN,
        "frozen_anchor_test_weighted_f1": frozen_anchor_f1,
        "finetuned_anchor_test_weighted_f1_by_seed": finetuned_anchor_f1,
        "frozen_full_r_test_weighted_f1_by_seed": {seed: frozen_full_r[seed]["test_weighted_f1"] for seed in SEEDS},
        "finetuned_full_r_test_weighted_f1_by_seed": {seed: finetuned_full_r[seed]["test_weighted_f1"] for seed in SEEDS},
        "frozen_paired_gain_by_seed": frozen_gain,
        "finetuned_paired_gain_by_seed": finetuned_gain,
        "finetuned_paired_gain_mean": finetuned_gain_mean,
        "finetuned_paired_gain_bootstrap_ci": [ft_ci_lo, ft_ci_hi],
        "finetuned_paired_gain_positive_and_excludes_zero": ft_excludes_zero_positive and finetuned_gain_mean > 0,
        "did_point_estimate": point,
        "did_ci": [ci_lo, ci_hi],
        "did_excludes_zero": excludes_zero,
        "did_frac_bootstrap_frozen_exceeds_finetuned": frac_frozen_exceeds,
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "verdict": verdict,
    }
    (DOCS_DIR / "deberta_large_did_result.json").write_text(json.dumps(result, indent=2))
    print(f"[compute_diff_in_diff_deberta_large] wrote docs/deberta_large_did_result.json", flush=True)


if __name__ == "__main__":
    main()
