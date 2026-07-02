"""Verify whether cached text features split into multiple inconsistent
random-projection spaces at crash/resume process boundaries (diagnostics only).

Identifies segment boundaries from file mtimes (a >5s gap between
consecutive writes indicates a new process), then compares within-segment
vs across-segment pairwise cosine similarity, plus a 2D PCA colored by
segment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

CACHE_DIR = Path("data/meld/cache/text")
GAP_THRESHOLD_SEC = 5.0


def find_segments() -> list[list[Path]]:
    files = list(CACHE_DIR.glob("*/*.pt"))
    files_with_mtime = sorted(((f.stat().st_mtime, f) for f in files), key=lambda x: x[0])

    segments: list[list[Path]] = [[]]
    prev_mtime = None
    for mtime, f in files_with_mtime:
        if prev_mtime is not None and (mtime - prev_mtime) > GAP_THRESHOLD_SEC:
            segments.append([])
        segments[-1].append(f)
        prev_mtime = mtime
    return segments


def load_vecs(paths: list[Path]) -> np.ndarray:
    return np.stack([torch.load(p, weights_only=True).numpy() for p in paths])


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def main() -> None:
    segments = find_segments()
    print(f"found {len(segments)} segment(s):")
    for i, seg in enumerate(segments):
        print(f"  segment {i}: {len(seg)} files")

    if len(segments) < 2:
        print("only one segment found — cannot test cross-segment inconsistency, nothing to compare.")
        return

    rng = np.random.default_rng(0)
    seg_a_sample_paths = [segments[0][i] for i in rng.choice(len(segments[0]), size=min(500, len(segments[0])), replace=False)]
    seg_a = load_vecs(seg_a_sample_paths)
    seg_b = load_vecs(segments[1])  # smallest/last segment, use all of it

    print(f"\nsegment A sample: {seg_a.shape}, segment B: {seg_b.shape}")

    # Within-segment-A pairwise cosine similarity (excluding diagonal)
    sim_aa = cosine_sim_matrix(seg_a, seg_a)
    iu = np.triu_indices_from(sim_aa, k=1)
    within_a = sim_aa[iu]
    print(f"\nwithin-segment-A cosine similarity: mean={within_a.mean():.4f} std={within_a.std():.4f} "
          f"min={within_a.min():.4f} max={within_a.max():.4f}")

    # Across-segment (A vs B) cosine similarity
    sim_ab = cosine_sim_matrix(seg_a, seg_b)
    across_ab = sim_ab.flatten()
    print(f"across segment A-vs-B cosine similarity: mean={across_ab.mean():.4f} std={across_ab.std():.4f} "
          f"min={across_ab.min():.4f} max={across_ab.max():.4f}")

    # z-score of the across-segment mean relative to the within-segment distribution
    z = (across_ab.mean() - within_a.mean()) / (within_a.std() + 1e-8)
    print(f"\nz-score of across-segment mean vs within-segment-A distribution: {z:.2f}")

    # 2D PCA, colored by segment
    pca = PCA(n_components=2, random_state=0)
    combined = np.concatenate([seg_a, seg_b], axis=0)
    proj = pca.fit_transform(combined)
    proj_a = proj[: len(seg_a)]
    proj_b = proj[len(seg_a) :]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(proj_a[:, 0], proj_a[:, 1], s=10, alpha=0.5, label=f"segment A (n={len(seg_a)})", color="#4C72B0")
    ax.scatter(proj_b[:, 0], proj_b[:, 1], s=120, marker="*", label=f"segment B (n={len(seg_b)})", color="#DD8452")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cached text features: 2D PCA by crash/resume segment")
    ax.legend()
    fig.tight_layout()
    out_path = Path("docs/text_segment_pca.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved PCA plot to {out_path}")
    print(f"segment A PC1/PC2 range: PC1=[{proj_a[:,0].min():.2f}, {proj_a[:,0].max():.2f}] "
          f"PC2=[{proj_a[:,1].min():.2f}, {proj_a[:,1].max():.2f}]")
    print(f"segment B PC1/PC2 values: {proj_b.tolist()}")


if __name__ == "__main__":
    main()
