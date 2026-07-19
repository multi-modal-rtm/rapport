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
COLOR_TEXT_ANCHOR = "#eda100"
COLOR_BASE_FUSION_R = "#2a78d6"
COLOR_FULL_R = "#1baf7a"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9})


def figure_a_k_sweep() -> None:
    data = json.loads((OUTPUTS_DIR / "subsumption_curve_data.json").read_text())
    text_anchor, base_fusion_r, full_r = data["text_anchor"], data["base_fusion_R"], data["full_R"]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    def plot_series(series: dict, color: str, label: str, marker: str):
        ks = K_VALUES
        means = [series[str(k)]["mean"] for k in ks]
        ax.plot(ks, means, color=color, linewidth=1.2, alpha=0.35, zorder=1)
        endpoint_ks = [k for k in ks if k in (0, 8)]
        endpoint_means = [series[str(k)]["mean"] for k in endpoint_ks]
        endpoint_errs = [series[str(k)]["std"] for k in endpoint_ks]
        endpoint_ns = [series[str(k)]["n_seeds"] for k in endpoint_ks]
        n_label = f"n={endpoint_ns[0]}" if len(set(endpoint_ns)) == 1 else "n=" + "/".join(f"{n}@k{k}" for k, n in zip(endpoint_ks, endpoint_ns))
        ax.errorbar(endpoint_ks, endpoint_means, yerr=endpoint_errs, color=color, fmt=marker, markersize=8, capsize=5, linewidth=1.8, label=f"{label} ({n_label})", zorder=3)
        interior_ks = [k for k in ks if k not in (0, 8)]
        interior_means = [series[str(k)]["mean"] for k in interior_ks]
        ax.scatter(interior_ks, interior_means, color=color, marker=marker, s=24, alpha=0.45, zorder=2)

    plot_series(text_anchor, COLOR_TEXT_ANCHOR, "Text anchor", "o")
    plot_series(base_fusion_r, COLOR_BASE_FUSION_R, "base_fusion_R", "s")
    plot_series(full_r, COLOR_FULL_R, "full_R", "^")

    ax.set_xlabel("Context window size $k$")
    ax.set_ylabel("Test weighted F1")
    ax.set_xticks(K_VALUES)
    ax.legend(loc="lower right", frameon=True)
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
    ax.set_xlabel("Edge update index (dialogue time)")
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
