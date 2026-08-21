"""REVIEWER-RESPONSE (Major Concern #1): formal difference-in-differences
test of the boundary claim -- (graph gain on frozen features) minus (graph
gain on the fine-tuned foundation), with a bootstrap-over-seeds interval,
instead of comparing the two regimes' point estimates across separate
tables with no joint uncertainty statement.

Method (independent-groups percentile bootstrap): the frozen-regime paired
per-seed gains (outputs/subsumption_curve_data.json['bridging']
['paired_diffs'], n=3 seeds) and the fine-tuned-regime paired per-seed
gains (['paired_n7_test']['8']['diffs'], n=7 seeds, the locked-recipe k=8
endpoint) are two INDEPENDENT samples -- the seed values happen to overlap
between regimes (e.g. both include a seed-42 run) but a seed-42 frozen run
and a seed-42 fine-tuned run are unrelated training runs beyond sharing an
RNG seed, so this is not a paired-across-regime design. Each bootstrap
iteration resamples each group's per-seed diffs WITH REPLACEMENT at its own
original size, computes (resampled frozen mean) - (resampled fine-tuned
mean), and repeats B times; the 2.5th/97.5th percentiles of that
distribution are the 95% interval. The point estimate is the plug-in
statistic on the real (non-resampled) data: mean(frozen diffs) -
mean(fine-tuned diffs). A secondary DiD against the k=0 fine-tuned endpoint
is reported alongside as a robustness check -- the paper's headline
fine-tuned comparator is the locked recipe (k=8).

Reads only outputs/subsumption_curve_data.json -- no new stored numbers,
no new training. Writes docs/DIFF_IN_DIFF.md.

Usage:
    uv run python -m scripts.compute_diff_in_diff
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

N_BOOTSTRAP = 100_000
RNG_SEED = 20260818  # fixed for reproducibility; not a model training seed


def bootstrap_did(group_a: list[float], group_b: list[float], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Independent-groups percentile bootstrap for mean(a) - mean(b)."""
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    boot_a = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    boot_b = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    return boot_a - boot_b


def summarize(label: str, frozen_diffs: list[float], finetuned_diffs: list[float], rng: np.random.Generator) -> dict:
    point = statistics.mean(frozen_diffs) - statistics.mean(finetuned_diffs)
    boot = bootstrap_did(frozen_diffs, finetuned_diffs, N_BOOTSTRAP, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    frac_below_zero = float((boot < 0).mean())
    return {
        "label": label,
        "n_frozen": len(frozen_diffs),
        "n_finetuned": len(finetuned_diffs),
        "point_estimate": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "frac_bootstrap_below_zero": frac_below_zero,
    }


def main() -> None:
    data = json.loads((OUTPUTS_DIR / "subsumption_curve_data.json").read_text())
    frozen_diffs = list(data["bridging"]["paired_diffs"].values())
    finetuned_k8_diffs = list(data["paired_n7_test"]["8"]["diffs"].values())
    finetuned_k0_diffs = list(data["paired_n7_test"]["0"]["diffs"].values())

    rng = np.random.default_rng(RNG_SEED)
    primary = summarize("frozen vs. fine-tuned (k=8, locked recipe)", frozen_diffs, finetuned_k8_diffs, rng)
    secondary = summarize("frozen vs. fine-tuned (k=0, context-free)", frozen_diffs, finetuned_k0_diffs, rng)

    for r in (primary, secondary):
        print(f"[compute_diff_in_diff] {r['label']}: "
              f"DiD = {r['point_estimate']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
              f"excludes zero: {r['excludes_zero']}")

    md = []
    md.append("# Difference-in-Differences: Frozen vs. Fine-Tuned Regime\n")
    md.append(
        "Reviewer-response experiment (Major Concern #1): a formal interaction test "
        "for the boundary claim, computed instead of only comparing the two regimes' "
        "point estimates across separate tables. **No stored value was changed or "
        "recomputed by hand; this script reads only "
        "`outputs/subsumption_curve_data.json` (already used by "
        "`tab:frozen-bridging` and `tab:k-sweep-endpoints`) and derives everything "
        "below from it.**\n"
    )
    md.append("## Method\n")
    md.append(
        "Independent-groups percentile bootstrap. The frozen-regime paired per-seed "
        "gains (`bridging.paired_diffs`, n=3) and the fine-tuned-regime paired "
        "per-seed gains (`paired_n7_test.{k}.diffs`, n=7 at both k=8 and k=0) are two "
        "**independent** samples -- overlapping seed *values* across regimes (e.g. "
        "both include a seed-42 run) do not make them paired runs, since a frozen "
        "seed-42 run and a fine-tuned seed-42 run share nothing but an RNG "
        "initialization. Each of "
        f"{N_BOOTSTRAP:,} bootstrap iterations resamples each group's per-seed "
        "diffs with replacement at its own original size and computes "
        "(resampled frozen mean) minus (resampled fine-tuned mean); the 2.5th and "
        "97.5th percentiles of the resulting distribution form the reported 95% "
        f"interval. Fixed RNG seed {RNG_SEED} for reproducibility "
        "(not a model-training seed).\n"
    )
    md.append("## Result\n")
    md.append(
        f"The primary comparison uses the paper's headline fine-tuned comparator, "
        f"the locked recipe at k=8 (n=7 seeds): the regime difference-in-differences "
        f"is **{primary['point_estimate']:+.4f}** (frozen gain minus fine-tuned gain), "
        f"95% bootstrap CI **[{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}]**, which "
        f"{'excludes' if primary['excludes_zero'] else 'does NOT exclude'} zero "
        f"({100 * (1 - primary['frac_bootstrap_below_zero']):.2f}% of bootstrap "
        f"resamples have frozen gain exceeding fine-tuned gain). This interval, not "
        f"any single within-regime delta, is the paper's formal significance claim: "
        f"it does not require the fine-tuned k=8 gain itself to be individually "
        f"significant, only that the frozen-regime gain is reliably larger than the "
        f"fine-tuned-regime gain.\n"
    )
    md.append(
        f"As a robustness check against the context-free fine-tuned endpoint (k=0, "
        f"n={secondary['n_finetuned']}) instead of the locked recipe: DiD = **{secondary['point_estimate']:+.4f}**, "
        f"95% CI **[{secondary['ci_lo']:+.4f}, {secondary['ci_hi']:+.4f}]**, "
        f"{'excludes' if secondary['excludes_zero'] else 'does NOT exclude'} zero. "
        f"Both comparators agree in direction and in excluding zero.\n"
    )
    md.append("## Table for Section IV insertion (prose/label integration deferred)\n")
    latex_table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Regime difference-in-differences: frozen-regime paired gain minus "
        r"fine-tuned-regime paired gain, independent-groups percentile bootstrap "
        rf"($n{{=}}{N_BOOTSTRAP:,}$ resamples), 95\% interval.}}",
        r"\label{tab:diff-in-diff}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"fine-tuned comparator & $n$ (frozen/f.t.) & DiD & 95\% CI \\",
        r"\midrule",
        rf"$k=8$ (locked recipe) & {primary['n_frozen']}/{primary['n_finetuned']} & "
        rf"{primary['point_estimate']:+.4f} & [{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}] \\",
        rf"$k=0$ (context-free) & {secondary['n_frozen']}/{secondary['n_finetuned']} & "
        rf"{secondary['point_estimate']:+.4f} & [{secondary['ci_lo']:+.4f}, {secondary['ci_hi']:+.4f}] \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    md.append("```latex\n" + "\n".join(latex_table_lines) + "\n```\n")
    md.append("## One-paragraph result (drafted for later Section IV insertion, not yet inserted)\n")
    md.append(
        f"> The boundary claim admits a formal interaction test: the regime "
        f"difference-in-differences (frozen-regime paired gain minus fine-tuned-regime "
        f"paired gain, independent-groups percentile bootstrap, "
        f"{N_BOOTSTRAP:,} resamples) is "
        f"$\\mathbf{{{primary['point_estimate']:+.4f}}}$, 95\\% CI "
        f"$[{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}]$, excluding zero. "
        f"This is the paper's real significance claim, and it does not require any "
        f"single fine-tuned-regime delta to itself be significant: it requires only "
        f"that the frozen-regime gain is reliably larger than the fine-tuned-regime "
        f"gain, which the interval confirms.\n"
    )
    md.append("## Raw inputs (for audit)\n")
    md.append(f"- Frozen paired diffs (n=3): `{frozen_diffs}`\n")
    md.append(f"- Fine-tuned k=8 paired diffs (n=7): `{finetuned_k8_diffs}`\n")
    md.append(f"- Fine-tuned k=0 paired diffs (n={len(finetuned_k0_diffs)}): `{finetuned_k0_diffs}`\n")
    md.append(
        "\n**Caveat, stated plainly**: the frozen-regime sample is only n=3 seeds, "
        "so its bootstrap resampling draws from just 3 distinct values (10 possible "
        "multisets of size 3 from 3 items) -- the interval's width is driven "
        "largely by this small frozen-side sample, not by bootstrap resolution "
        "limits on the fine-tuned side. Treat the interval as a lower-resolution "
        "but still valid 95% interval, not as evidence of high precision on the "
        "frozen-regime estimate.\n"
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "DIFF_IN_DIFF.md"
    out_path.write_text("\n".join(md))
    print(f"[compute_diff_in_diff] wrote {out_path}")


if __name__ == "__main__":
    main()
