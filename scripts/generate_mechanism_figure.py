"""PAPER-ASSETS PHASE, figure (d): the 4.5 mechanism -- scratch (N4)
retraining vs. residual (N4-R) attribution, MELD, fine-tuned foundation.
Design: two panels side by side (scratch | residual), each a small
point-range chart with the text anchor as a horizontal reference line and
base_fusion/full (or their _R counterparts) as two points connected by a
bracket showing the gap -- visualizes "the gap shrinks by ~63% once the
incidental-presence confound is removed," matching Section 4.5's claim.

Reads only outputs/*/metrics.json -- no hand-typed numbers.

Usage:
    uv run python -m scripts.generate_mechanism_figure
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
FIGURES_DIR = PROJECT_ROOT / "paper_assets" / "figures"

SEEDS = (42, 1337, 2024)
COLOR_BASELINE = "#2a78d6"  # slot1 blue
COLOR_FULL = "#1baf7a"  # slot2 aqua
COLOR_ANCHOR = "#eda100"  # slot3 yellow


def load(name: str) -> dict:
    return json.loads((OUTPUTS_DIR / name / "metrics.json").read_text())


def mean_std(vals: list[float]) -> tuple[float, float]:
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    anchor_vals = [load(f"context_text_plain_ce_seed{s}")["test_weighted_f1"] for s in SEEDS]
    anchor_m, anchor_s = mean_std(anchor_vals)

    scratch_base = mean_std([load(f"base_fusion_seed{s}")["test_weighted_f1"] for s in SEEDS])
    scratch_full = mean_std([load(f"full_seed{s}")["test_weighted_f1"] for s in SEEDS])
    residual_base = mean_std([load(f"base_fusion_R_seed{s}")["test_weighted_f1"] for s in SEEDS])
    residual_full = mean_std([load(f"full_R_seed{s}")["test_weighted_f1"] for s in SEEDS])

    scratch_gap = scratch_full[0] - scratch_base[0]
    residual_gap = residual_full[0] - residual_base[0]
    shrinkage_pct = (1 - abs(residual_gap) / abs(scratch_gap)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5), sharey=True)

    panel_data = [
        ("Scratch retraining (N4)", scratch_base, scratch_full, scratch_gap, axes[0]),
        ("Residual attribution (N4-R)", residual_base, residual_full, residual_gap, axes[1]),
    ]

    for title, base, full, gap, ax in panel_data:
        ax.axhspan(anchor_m - anchor_s, anchor_m + anchor_s, color=COLOR_ANCHOR, alpha=0.15, zorder=0)
        ax.axhline(anchor_m, color=COLOR_ANCHOR, linewidth=1.5, linestyle="--", label=f"text anchor ({anchor_m:.4f})", zorder=1)

        ax.errorbar(0, base[0], yerr=base[1], color=COLOR_BASELINE, fmt="s", markersize=11, capsize=6, linewidth=2, label="baseline (fusion only)", zorder=3)
        ax.errorbar(1, full[0], yerr=full[1], color=COLOR_FULL, fmt="^", markersize=11, capsize=6, linewidth=2, label="full stack", zorder=3)
        ax.plot([0, 1], [base[0], full[0]], color="#8a8a86", linewidth=1.2, linestyle=":", zorder=2)
        ax.annotate(f"gap: {gap:+.4f}", xy=(0.5, (base[0] + full[0]) / 2), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", "full stack"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Test weighted F1")
    axes[1].legend(loc="lower left", fontsize=8, frameon=True)
    fig.suptitle(
        f"MELD mechanism: scratch retraining vs. residual attribution\n"
        f"(baseline$\\to$full gap shrinks {shrinkage_pct:.0f}\\% under residual attribution)",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig_path = FIGURES_DIR / f"mechanism_figure.{ext}"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300 if ext == "png" else None)
        print(f"[generate_mechanism_figure] wrote {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
