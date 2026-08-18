"""PAPER-ASSETS PHASE, figures (a) and (c): publication-quality (vector
PDF, colorblind-safe reference palette, single-column-legible font sizes)
re-renders of the k-sweep curve and the IEMOCAP edge-state trajectory.
Reads the SAME stored JSON data the original docs/ PNG figures were built
from (outputs/subsumption_curve_data.json,
outputs/iemocap_edge_state_trajectory_data.json) -- no new computation, no
hand-typed numbers, just a higher-fidelity render for camera-ready use.

Usage:
    uv run python -m scripts.generate_publication_figures
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

K_VALUES = [0, 2, 4, 8]
COLOR_GAIN = "#2a78d6"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9})


def figure_a_k_sweep() -> None:
    # Plots the paired per-seed gain (Full stack minus its own Text-only
    # anchor) at each context width, matching what the caption and body
    # text describe and what tab:k-sweep-endpoints reports -- not the raw
    # per-config weighted F1. Endpoints (k=0, k=8) use the multi-seed
    # paired series (paired_n7_test, n=7/n=7) with error bars; interior
    # points (k=2, k=4) use the single-seed gain_table entries as trend
    # indicators only, de-emphasized (no error bar, lighter marker).
    data = json.loads((OUTPUTS_DIR / "subsumption_curve_data.json").read_text())
    paired = data["paired_n7_test"]
    gain_table = {row["k"]: row for row in data["gain_table"]}

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    endpoint_ks = [0, 8]
    endpoint_means = [paired[str(k)]["mean"] for k in endpoint_ks]
    endpoint_errs = [paired[str(k)]["std"] for k in endpoint_ks]
    endpoint_ns = [paired[str(k)]["n_pairs"] for k in endpoint_ks]

    interior_ks = [k for k in K_VALUES if k not in endpoint_ks]
    interior_means = [gain_table[k]["gain"] for k in interior_ks]

    all_ks = sorted(endpoint_ks + interior_ks)
    all_means = [paired[str(k)]["mean"] if k in endpoint_ks else gain_table[k]["gain"] for k in all_ks]
    ax.plot(all_ks, all_means, color=COLOR_GAIN, linewidth=1.2, alpha=0.35, zorder=1)

    ax.scatter(interior_ks, interior_means, color=COLOR_GAIN, marker="o", s=28, alpha=0.45, zorder=2,
               label="Interior points (single seed, trend only)")
    endpoint_n_label = "/".join(f"n={n}" for n in endpoint_ns)
    ax.errorbar(endpoint_ks, endpoint_means, yerr=endpoint_errs, color=COLOR_GAIN, fmt="s", markersize=9,
                capsize=6, linewidth=2, zorder=3, label=f"Endpoints ({endpoint_n_label} seeds)")

    ax.axhline(0, color="#8a8a86", linewidth=1.2, linestyle="--", zorder=0)
    ax.set_xlabel("Context window size $k$")
    ax.set_ylabel("Paired gain in weighted F1")
    ax.set_xticks(K_VALUES)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"k_sweep_curve.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"[generate_publication_figures] wrote {path}")
    plt.close(fig)


def figure_c_edge_state_trajectory() -> None:
    data = json.loads((OUTPUTS_DIR / "iemocap_edge_state_trajectory_data.json").read_text())
    colors = ["#2a78d6", "#1baf7a", "#eda100"]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for i, (did, norms) in enumerate(data.items()):
        ax.plot(range(1, len(norms) + 1), norms, color=colors[i % len(colors)], marker="o", markersize=3, linewidth=1.2, label=f"dialogue {did} ($n$={len(norms)})")
    ax.set_xlabel("Edge update index (dialogue time, one per speaking turn)")
    ax.set_ylabel("Edge-state $\\|e_{sj}\\|$ (L2 norm)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"edge_state_trajectory.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"[generate_publication_figures] wrote {path}")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_a_k_sweep()
    figure_c_edge_state_trajectory()


if __name__ == "__main__":
    main()
