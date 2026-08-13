"""PAPER-ASSETS PHASE, figure (b): the frozen-vs-fine-tuned paired-gain
contrast (the boundary figure). Design: a horizontal point-range
("dumbbell"-style, one point per regime rather than two connected) chart
-- one row per regime, x-axis is the paired gain (full_R minus its own
anchor), a zero reference line, points colored by sign (diverging
blue/red per the dataviz skill's reference palette, not the categorical
identity colors used elsewhere in this project's figures, since this
chart's job is POLARITY not series identity).

Reads only outputs/subsumption_curve_data.json (already itself derived
from stored metrics.json files) -- no hand-typed numbers.

Usage:
    uv run python -m scripts.generate_boundary_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "paper_assets" / "figures"

COLOR_POSITIVE = "#2a78d6"  # diverging pair: blue pole (skill reference palette)
COLOR_NEGATIVE = "#e34948"  # diverging pair: red pole


def main() -> None:
    data = json.loads((OUTPUTS_DIR / "subsumption_curve_data.json").read_text())
    bridging = data["bridging"]
    paired = data["paired_n7_test"]

    # marker shape encodes frozen-vs-fine-tuned identity (not just color/sign),
    # so the two fine-tuned endpoints are distinguishable from the frozen one
    # without relying on color alone.
    rows = [
        ("Frozen foundation\n(bridging experiment)", bridging["paired_mean"], bridging["paired_std"], bridging["full_R_frozen"]["n_seeds"], "o"),
        ("Fine-tuned foundation, $k=0$", paired["0"]["mean"], paired["0"]["std"], paired["0"]["n_pairs"], "s"),
        ("Fine-tuned foundation, $k=8$\n(locked recipe)", paired["8"]["mean"], paired["8"]["std"], paired["8"]["n_pairs"], "s"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_positions = list(range(len(rows)))[::-1]

    for y, (label, mean, std, n, marker) in zip(y_positions, rows):
        color = COLOR_POSITIVE if mean > 0 else COLOR_NEGATIVE
        ax.errorbar(mean, y, xerr=std, color=color, fmt=marker, markersize=11, capsize=6, linewidth=2, zorder=3)
        sign_label = f"{mean:+.4f} $\\pm$ {std:.4f} (n={n})"
        ax.annotate(
            sign_label, xy=(mean, y), xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=9, color="#2b2b28",
        )

    ax.axvline(0, color="#8a8a86", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("Paired per-seed gain: Full stack $-$ Text-only anchor (test weighted F1)")
    ax.set_title("The fine-tuning boundary: graph machinery's paired gain,\nfrozen vs. fine-tuned foundations (MELD)", fontsize=11)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig_path = FIGURES_DIR / f"boundary_figure.{ext}"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300 if ext == "png" else None)
        print(f"[generate_boundary_figure] wrote {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
