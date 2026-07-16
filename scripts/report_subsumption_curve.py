"""Phase N5-A: the subsumption curve. Reads all text-encoder and
base_fusion_R/full_R metrics.json files across k in {0, 2, 4, 8} and
produces the centerpiece figure (docs/subsumption_curve.png) plus the
marginal-gain table (docs/PHASE_N5A.md).

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
ENDPOINT_SEEDS = (42, 1337, 2024)

# Palette (dataviz skill reference palette, light mode, fixed categorical order).
COLOR_TEXT_ANCHOR = "#eda100"  # slot 3, yellow
COLOR_BASE_FUSION_R = "#2a78d6"  # slot 1, blue
COLOR_FULL_R = "#1baf7a"  # slot 2, aqua


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


def main() -> None:
    text_anchor: dict[int, dict] = {}
    base_fusion_r: dict[int, dict] = {}
    full_r: dict[int, dict] = {}

    for k in K_VALUES:
        rapport_seeds = ENDPOINT_SEEDS if k in (0, 8) else (42,)
        # The text encoder (Step 1) is seed-42-only at EVERY k, per the phase spec --
        # k=8 is the one exception, where we reuse Phase T's pre-existing 3-seed data
        # (context_text_plain_ce_seed{42,1337,2024}) rather than "retraining" it.
        anchor_seeds = ENDPOINT_SEEDS if k == 8 else (42,)

        anchor_vals = [load(text_anchor_run_name(k, s))["test_weighted_f1"] for s in anchor_seeds]
        anchor_mean, anchor_std = mean_std(anchor_vals)
        text_anchor[k] = {"mean": anchor_mean, "std": anchor_std, "values": anchor_vals, "n_seeds": len(anchor_seeds)}

        base_vals = [load(rapport_run_name("base_fusion_R", k, s))["test_weighted_f1"] for s in rapport_seeds]
        base_mean, base_std = mean_std(base_vals)
        base_fusion_r[k] = {"mean": base_mean, "std": base_std, "values": base_vals, "n_seeds": len(rapport_seeds)}

        full_vals = [load(rapport_run_name("full_R", k, s))["test_weighted_f1"] for s in rapport_seeds]
        full_mean, full_std = mean_std(full_vals)
        full_r[k] = {"mean": full_mean, "std": full_std, "values": full_vals, "n_seeds": len(rapport_seeds)}

    # ---- Marginal gain table: full_R - base_fusion_R ----
    gain_table = []
    for k in K_VALUES:
        gain_mean = full_r[k]["mean"] - base_fusion_r[k]["mean"]
        gain_table.append({"k": k, "gain": gain_mean, "n_seeds": full_r[k]["n_seeds"]})

    # Pre-registered check: monotonically shrinking gain with k?
    gains = [row["gain"] for row in gain_table]
    strictly_monotonic_decreasing = all(gains[i] > gains[i + 1] for i in range(len(gains) - 1))
    endpoints_only_decreasing = gains[0] > gains[-1]

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(8, 5.5))

    def plot_series(data: dict[int, dict], color: str, label: str, marker: str):
        ks = K_VALUES
        means = [data[k]["mean"] for k in ks]
        errs = [data[k]["std"] if data[k]["n_seeds"] > 1 else 0.0 for k in ks]
        ax.plot(ks, means, color=color, marker=marker, markersize=7, linewidth=2, label=label, zorder=3)
        ax.errorbar(ks, means, yerr=errs, color=color, fmt="none", capsize=4, linewidth=1.5, zorder=2)

    plot_series(text_anchor, COLOR_TEXT_ANCHOR, "Text anchor (Phase T, trained at k)", "o")
    plot_series(base_fusion_r, COLOR_BASE_FUSION_R, "base_fusion_R", "s")
    plot_series(full_r, COLOR_FULL_R, "full_R", "^")

    ax.set_xlabel("Context window size k (utterances of preceding history)")
    ax.set_ylabel("Test weighted F1 (raw)")
    ax.set_title("MELD subsumption curve: what the graph adds, by context window size")
    ax.set_xticks(K_VALUES)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig_path = DOCS_DIR / "subsumption_curve.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- Write docs/PHASE_N5A.md ----
    lines = []
    lines.append("# PHASE N5-A — the subsumption curve\n")
    lines.append(f"![subsumption curve]({fig_path.name})\n")
    lines.append("## Data (mean ± std; single point where n_seeds=1)\n")
    lines.append("| k | text anchor | base_fusion_R | full_R | n_seeds (base_fusion_R/full_R) |")
    lines.append("|---|---|---|---|---|")
    for k in K_VALUES:
        ta, bf, fr = text_anchor[k], base_fusion_r[k], full_r[k]
        lines.append(
            f"| {k} | {ta['mean']:.4f}±{ta['std']:.4f} | {bf['mean']:.4f}±{bf['std']:.4f} | "
            f"{fr['mean']:.4f}±{fr['std']:.4f} | {bf['n_seeds']} |"
        )
    lines.append("")

    lines.append("## Graph's marginal gain per k (full_R − base_fusion_R)\n")
    lines.append("| k | gain | n_seeds |")
    lines.append("|---|---|---|")
    for row in gain_table:
        lines.append(f"| {row['k']} | {row['gain']:+.4f} | {row['n_seeds']} |")
    lines.append("")

    lines.append("## Pre-registered reading: monotonically shrinking gain with k?\n")
    lines.append(f"- Strictly monotonic decreasing across all 4 points: **{strictly_monotonic_decreasing}**")
    lines.append(
        f"- Endpoints only (k=0, robust 3-seed mean, vs. k=8, robust 3-seed mean): "
        f"gain shrinks from {gains[0]:+.4f} to {gains[-1]:+.4f} -> **{endpoints_only_decreasing}**"
    )
    lines.append("")

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
    }
    (OUTPUTS_DIR / "subsumption_curve_data.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
