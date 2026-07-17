"""Phase N5-A (+ consolidation): the subsumption curve. Reads all
text-encoder and base_fusion_R/full_R metrics.json files across k in
{0, 2, 4, 8} and produces the centerpiece figure
(docs/subsumption_curve.png) plus docs/PHASE_N5A.md in full (including the
static reconciliation section below, which is authored content, not
derived from data, but kept here so the whole doc regenerates from one
script per its own Reproducibility promise).

n=7 consolidation: k=0/k=8 endpoints use 7 seeds (42, 1337, 2024, 7, 123,
555, 9090) for full_R ("the graph config") at both endpoints and for the
text anchor at k=8. The k=0 text anchor was seed-42-only in the original
design and only gained the 4 new seeds {7,123,555,9090} here, so it's
n=5, not n=7 (see K0_TEXT_ANCHOR_SEEDS). base_fusion_R stays at 3 seeds
(not powered up); k=2/k=4 remain single-seed (42 only).

Usage:
    uv run python -m scripts.report_subsumption_curve
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

K_VALUES = [0, 2, 4, 8]
BASE_FUSION_SEEDS = (42, 1337, 2024)  # not powered up
N7_SEEDS = (42, 1337, 2024, 7, 123, 555, 9090)  # full_R at both endpoints; text anchor at k=8
# k=0 text anchor was seed-42-only in the original design (Step 1 scoped the
# text-encoder retrain to seed 42 at every k except k=8). The consolidation
# task added exactly 4 new seeds {7,123,555,9090} -- not 6 -- so k=0 text
# anchor lands at n=5, not n=7. Only full_R (all downstream reruns on the
# existing shared per-k cache) and the k=8 text anchor (which already had
# 3 genuine encoder retrains from Phase T) reach n=7.
K0_TEXT_ANCHOR_SEEDS = (42, 7, 123, 555, 9090)

# Palette (dataviz skill reference palette, light mode, fixed categorical order).
COLOR_TEXT_ANCHOR = "#eda100"  # slot 3, yellow
COLOR_BASE_FUSION_R = "#2a78d6"  # slot 1, blue
COLOR_FULL_R = "#1baf7a"  # slot 2, aqua

RECONCILIATION_SECTION = """## Reconciliation: N4-R Step 2's +0.0147 vs. this phase's +0.0025 at k=0

These are **not two measurements of the same quantity** -- every one of
four things differs between them:

| | N4-R Step 2 (`docs/PHASE_N4R.md`) | N5-A (this doc) |
|---|---|---|
| **Architecture** | `base_fusion` -- the ORIGINAL, pre-residual Phase N4 architecture (`residual` flag didn't exist yet; Step 2 ran before Step 3's redesign) | `base_fusion_R` / `full_R` -- the residual redesign (spec v1.1: zero-init `W_out`, zero-init fusion A/V blocks) |
| **k=0 text encoder** | The SAME k=8-TRAINED Phase T checkpoint (`context_text_plain_ce_seed42`), fed context-free input only AT INFERENCE TIME when building the cache -- never retrained | A GENUINELY freshly fine-tuned encoder, LoRA-trained end-to-end with k=0 context windows during its OWN training (`context_text_k0_seed42`) |
| **Gain definition** | `base_fusion` (which already contains the v1-style per-speaker GAT+GRU -- there is no "graph-free" baseline in the original architecture) **minus** the text-only anchor | `full_R` (relational memory + shift + temporal, on top of `base_fusion_R`'s own GAT+GRU) **minus** `base_fusion_R` (fusion + GAT+GRU alone, no relational memory/shift/temporal) |
| **Seeds** | 1 (seed 42) | 3, now 7 at the endpoints (seeds 42/1337/2024/7/123/555/9090) |

**They measure different things by design, not by accident.** Step 2 was
an explicitly-labeled "one cheap run" triage diagnostic (Phase N4-R's own
words) to decide *whether the residual redesign was worth building at
all* -- it asked "does the whole fusion+graph stack beat text alone more
when text has no context of its own," using whatever encoder was already
on hand (inference-time context removal, not a real retrain, since
retraining a whole new encoder wasn't yet justified at that stage). N5-A
asks the narrower, better-controlled question that Phase N4-R's own
conclusion motivated: "does the INCREMENTAL relational/shift/temporal
machinery earn its keep over a matched fusion-only baseline, when the
encoder itself was genuinely trained to have no cross-utterance context" --
using the current (post-redesign) architecture throughout.

**For completeness, the literal analog of Step 2's definition recomputed
on N5-A's assets** (`base_fusion_R` minus the genuinely-k0-trained text
anchor, at k=0, 3-seed `base_fusion_R` mean vs. seed-42 text anchor):
0.6279 - 0.6395 = **-0.0116** -- the opposite sign from Step 2's +0.0147.
This is a real, informative difference, not noise: it shows that once the
encoder is actually retrained at k=0 (rather than just fed k=0 input
through k=8-trained weights), the "text-only" anchor itself is
considerably stronger (0.6395 vs. Step 2's inference-only anchor of
0.6234) -- strong enough that `base_fusion_R`'s fusion+graph stack no
longer beats it at all. Step 2's inference-only anchor understated how
good a properly-trained context-free encoder actually is.

**Which number the paper reports: N5-A's (the n=7 `full_R` vs. text-anchor
paired comparison below), not Step 2's.** N5-A uses the current
architecture (the one Phase N4-R actually shipped), a methodologically
correct encoder construction (genuinely trained at k=0, not
inference-time-perturbed), and multiple seeds rather than 1. Step 2's
number stays on the record as the preliminary diagnostic that motivated
building N5-A properly, not as a competing or superseded-by-noise result
-- it was never intended to be the paper's statistic.
"""


def load(run_name: str) -> dict:
    return json.loads((OUTPUTS_DIR / run_name / "metrics.json").read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def text_anchor_run_name(k: int, seed: int) -> str:
    if k == 8:
        return f"context_text_plain_ce_seed{seed}"
    return f"context_text_k{k}_seed{seed}"


def rapport_run_name(config: str, k: int, seed: int) -> str:
    if k == 8:
        return f"{config}_seed{seed}"
    return f"{config}_k{k}_seed{seed}"


def series_at_k(config_or_anchor: str, k: int, seeds: tuple[int, ...]) -> dict:
    if config_or_anchor == "text_anchor":
        vals = {s: load(text_anchor_run_name(k, s))["test_weighted_f1"] for s in seeds}
    else:
        vals = {s: load(rapport_run_name(config_or_anchor, k, s))["test_weighted_f1"] for s in seeds}
    mean, std = mean_std(list(vals.values()))
    return {"mean": mean, "std": std, "values": vals, "n_seeds": len(seeds)}


def main() -> None:
    text_anchor: dict[int, dict] = {}
    base_fusion_r: dict[int, dict] = {}
    full_r: dict[int, dict] = {}

    for k in K_VALUES:
        endpoint = k in (0, 8)
        if not endpoint:
            anchor_seeds = (42,)
        elif k == 0:
            anchor_seeds = K0_TEXT_ANCHOR_SEEDS
        else:
            anchor_seeds = N7_SEEDS
        rapport_seeds_base = BASE_FUSION_SEEDS if endpoint else (42,)
        rapport_seeds_full = N7_SEEDS if endpoint else (42,)  # full_R IS the powered-up "graph config"

        text_anchor[k] = series_at_k("text_anchor", k, anchor_seeds)
        base_fusion_r[k] = series_at_k("base_fusion_R", k, rapport_seeds_base)
        full_r[k] = series_at_k("full_R", k, rapport_seeds_full)

    # ---- Graph's marginal gain per k (full_R - base_fusion_R), unpaired means ----
    gain_table = []
    for k in K_VALUES:
        gain_mean = full_r[k]["mean"] - base_fusion_r[k]["mean"]
        gain_table.append({"k": k, "gain": gain_mean, "n_seeds": full_r[k]["n_seeds"]})
    gains = [row["gain"] for row in gain_table]
    strictly_monotonic_decreasing = all(gains[i] > gains[i + 1] for i in range(len(gains) - 1))
    endpoints_only_decreasing = gains[0] > gains[-1]

    # ---- PRE-REGISTERED test: paired (per-seed) full_R minus text-anchor, at k=0 and k=8.
    # Paired over the INTERSECTION of seeds present in both series -- at k=8
    # that's the full N7_SEEDS (7 pairs); at k=0 the text anchor only has 5
    # distinct encoder retrains (K0_TEXT_ANCHOR_SEEDS), so seeds 1337/2024
    # (which have a full_R downstream run but no matching k=0 text-anchor
    # retrain under that seed label) are excluded rather than paired against
    # a different seed's anchor value. This yields n=5 pairs at k=0, n=7 at k=8.
    paired = {}
    for k in (0, 8):
        pair_seeds = sorted(set(full_r[k]["values"]) & set(text_anchor[k]["values"]))
        diffs = {s: full_r[k]["values"][s] - text_anchor[k]["values"][s] for s in pair_seeds}
        diff_mean, diff_std = mean_std(list(diffs.values()))
        paired[k] = {"diffs": diffs, "mean": diff_mean, "std": diff_std, "n_pairs": len(pair_seeds)}

    k0_gain = paired[0]["mean"]
    k0_std = paired[0]["std"]
    subsumption_confirmed = k0_gain > 0 and abs(k0_gain) > k0_std

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(8.5, 6))

    def plot_series(data: dict[int, dict], color: str, label: str, marker: str):
        ks = K_VALUES
        means = [data[k]["mean"] for k in ks]
        # full connecting line, de-emphasized (the interior k=2/4 segments are single-seed)
        ax.plot(ks, means, color=color, linewidth=1.5, alpha=0.35, zorder=1)

        # bold endpoints (k=0, k=8) with error bars
        endpoint_ks = [k for k in ks if k in (0, 8)]
        endpoint_means = [data[k]["mean"] for k in endpoint_ks]
        endpoint_errs = [data[k]["std"] for k in endpoint_ks]
        endpoint_ns = [data[k]["n_seeds"] for k in endpoint_ks]
        n_label = f"n={endpoint_ns[0]}" if len(set(endpoint_ns)) == 1 else "n=" + "/".join(
            f"{n}@k{k}" for k, n in zip(endpoint_ks, endpoint_ns)
        )
        ax.errorbar(
            endpoint_ks, endpoint_means, yerr=endpoint_errs, color=color, fmt=marker, markersize=9,
            capsize=5, linewidth=2, label=f"{label} ({n_label})", zorder=3,
        )

        # de-emphasized interior points (k=2, k=4): smaller, paler, no error bars (n=1)
        interior_ks = [k for k in ks if k not in (0, 8)]
        interior_means = [data[k]["mean"] for k in interior_ks]
        ax.scatter(interior_ks, interior_means, color=color, marker=marker, s=28, alpha=0.45, zorder=2)

    plot_series(text_anchor, COLOR_TEXT_ANCHOR, "Text anchor (Phase T, trained at k)", "o")
    plot_series(base_fusion_r, COLOR_BASE_FUSION_R, "base_fusion_R", "s")
    plot_series(full_r, COLOR_FULL_R, "full_R", "^")

    ax.set_xlabel("Context window size k (utterances of preceding history)")
    ax.set_ylabel("Test weighted F1 (raw)")
    ax.set_title("MELD subsumption curve: what the graph adds, by context window size\n(bold = multi-seed endpoints, see legend for n; pale = single-seed interior points)", fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig_path = DOCS_DIR / "subsumption_curve.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- Write docs/PHASE_N5A.md ----
    lines = []
    lines.append("# PHASE N5-A — the subsumption curve (MELD)\n")
    lines.append(
        "Chart of context-window size k vs. what the relational/shift/temporal "
        "stack adds on top of the residual-redesigned fusion baseline "
        "(`docs/PHASE_N4R.md` spec v1.1). For k in {0, 2, 4, 8}: (1) retrained the "
        "Phase T text encoder end-to-end AT that k, caching its embeddings + own "
        "logits (`cache_version text_ctx_k{k}`); (2) trained `base_fusion_R` and "
        "`full_R` on that cache; (3) at the endpoints k=0 and k=8, `full_R` "
        "(\"the graph config\") was powered up to **n=7 seeds** at BOTH endpoints "
        "(42, 1337, 2024, 7, 123, 555, 9090) -- all downstream reruns on the same "
        "shared per-k cache, per the original 3-seed convention. The text anchor "
        "reaches n=7 at k=8 (Phase T's existing 3 seeds + 4 new retrains) but only "
        "**n=5 at k=0** (42, 7, 123, 555, 9090): the k=0 text anchor was seed-42-only "
        "in the original design, and this consolidation pass added exactly the 4 "
        "new seeds it was asked to add, not 6. `base_fusion_R` stays at n=3 "
        "(seeds 42/1337/2024, not powered up in this consolidation pass).\n"
    )
    lines.append(fig_path.name and f"![subsumption curve]({fig_path.name})\n")
    lines.append(RECONCILIATION_SECTION)

    lines.append("## Data (mean ± std; single point where n_seeds=1)\n")
    lines.append("| k | text anchor | base_fusion_R | full_R | n_seeds (anchor/full_R, base_fusion_R) |")
    lines.append("|---|---|---|---|---|")
    for k in K_VALUES:
        ta, bf, fr = text_anchor[k], base_fusion_r[k], full_r[k]
        lines.append(
            f"| {k} | {ta['mean']:.4f}±{ta['std']:.4f} | {bf['mean']:.4f}±{bf['std']:.4f} | "
            f"{fr['mean']:.4f}±{fr['std']:.4f} | {ta['n_seeds']}/{fr['n_seeds']}, {bf['n_seeds']} |"
        )
    lines.append("")

    lines.append("## Graph's marginal gain per k (full_R − base_fusion_R, unpaired means)\n")
    lines.append("| k | gain | n_seeds (full_R) |")
    lines.append("|---|---|---|")
    for row in gain_table:
        lines.append(f"| {row['k']} | {row['gain']:+.4f} | {row['n_seeds']} |")
    lines.append("")

    lines.append("## PRE-REGISTERED test: paired (per-seed) full_R − text anchor, at the endpoints\n")
    lines.append(
        "Paired over the intersection of seeds present in both series. At k=8 "
        "both series have all 7 seeds, so n_pairs=7. At k=0 the text anchor only "
        "has 5 distinct encoder retrains (see above), so seeds 1337 and 2024 -- "
        "which have a `full_R` downstream run but no matching k=0 text-anchor "
        "retrain under that seed label -- are excluded rather than paired against "
        "a different seed's anchor value, giving n_pairs=5 at k=0.\n"
    )
    lines.append("| k | n_pairs | per-seed diffs (seed: diff) | mean | std | |diff|>std? |")
    lines.append("|---|---|---|---|---|---|")
    for k in (0, 8):
        p = paired[k]
        diffs_str = ", ".join(f"{s}:{d:+.4f}" for s, d in p["diffs"].items())
        lines.append(
            f"| {k} | {p['n_pairs']} | {diffs_str} | {p['mean']:+.4f} | {p['std']:.4f} | "
            f"{abs(p['mean']) > p['std']} |"
        )
    lines.append("")

    lines.append("## PRE-REGISTERED DECISION (fixed before this result was computed)\n")
    lines.append(
        f"Decided on the k=0 paired result at n_pairs={paired[0]['n_pairs']} (5, not 7 -- see the "
        f"paired-test note above on why 1337/2024 are excluded at k=0).\n"
    )
    if subsumption_confirmed:
        lines.append(
            f"**k=0 paired gain is positive ({k0_gain:+.4f}) and |gain| ({abs(k0_gain):.4f}) > std "
            f"({k0_std:.4f}): SUBSUMPTION NARRATIVE CONFIRMED.** The graph (relational memory + "
            f"shift + temporal) measurably helps a context-free text encoder (k=0), and "
            f"docs/PHASE_N4R.md / this same table at k=8 already show that advantage vanishing "
            f"once the encoder has its own k=8 context -- contextual encoding substitutes for "
            f"graph-level recency signal, exactly as the paper's subsumption claim states."
        )
    else:
        lines.append(
            f"**k=0 paired gain ({k0_gain:+.4f}) is within noise (std {k0_std:.4f}): THE STRONGER "
            f"NULL.** Even in the one condition most favorable to the graph -- a text encoder with "
            f"little to no context of its own -- `full_R` does not measurably beat the text-only "
            f"anchor. The paper's claim should be framed accordingly: the relational/shift/temporal "
            f"machinery does not demonstrate a measurable benefit over a strong text baseline at "
            f"ANY tested context window size, not just at the locked k=8 recipe."
        )
    lines.append("")

    lines.append("## Legacy: monotonicity of the unpaired full_R − base_fusion_R gain\n")
    lines.append(f"- Strictly monotonic decreasing across all 4 points: **{strictly_monotonic_decreasing}**")
    lines.append(
        f"- Endpoints only (k=0 vs k=8, both robust means): gain shrinks from "
        f"{gains[0]:+.4f} to {gains[-1]:+.4f} -> **{endpoints_only_decreasing}**"
    )
    lines.append("")

    lines.append("## Reproducibility\n")
    lines.append(
        "`scripts/report_subsumption_curve.py` regenerates this ENTIRE doc (including the "
        "reconciliation section above, which is static authored text kept in the script) and the "
        "figure from `outputs/*/metrics.json` directly -- rerun any time after "
        "`scripts/run_subsumption_matrix.py` / `scripts/run_n7_powerup.py`. "
        "`outputs/subsumption_curve_data.json` has the full underlying per-seed data."
    )

    report_text = "\n".join(lines)
    (DOCS_DIR / "PHASE_N5A.md").write_text(report_text)
    print(f"[done] wrote {fig_path} and {DOCS_DIR / 'PHASE_N5A.md'}")
    print(report_text)

    summary = {
        "text_anchor": text_anchor,
        "base_fusion_R": base_fusion_r,
        "full_R": full_r,
        "gain_table": gain_table,
        "strictly_monotonic_decreasing": strictly_monotonic_decreasing,
        "endpoints_only_decreasing": endpoints_only_decreasing,
        "paired_n7_test": paired,
        "subsumption_confirmed": subsumption_confirmed,
    }
    (OUTPUTS_DIR / "subsumption_curve_data.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
