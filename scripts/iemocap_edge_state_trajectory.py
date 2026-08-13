"""Phase N5-B Step B3.2: dyadic qualitative artifact -- logs and plots the
single edge-state norm over dialogue time for 3 sample Session5 (test
split) dialogues, using the trained full_R_iemocap_seed42 checkpoint.
IEMOCAP dialogues are strictly dyadic (2 speakers), so there is exactly
ONE edge (undirected pair) per dialogue to track -- `RelationalEdgeMemory
.update_incident_edges` is monkeypatched (not modified) to record the
edge state's L2 norm each time it's called, non-invasively, using the
model's own real, tested forward pass (RapportModel._forward_relational)
with no changes to any model source file.

Usage:
    uv run python -m scripts.iemocap_edge_state_trajectory
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.models.rapport_model import RapportModel
from rapport.models.relational_memory import RelationalEdgeMemory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "iemocap" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "iemocap" / "cache"
DOCS_DIR = PROJECT_ROOT / "docs"
NUM_CLASSES = 6
CHECKPOINT = PROJECT_ROOT / "outputs" / "full_R_iemocap_seed42" / "best_model.pt"

# Reference palette (dataviz skill, light mode, fixed categorical order).
COLORS = ["#2a78d6", "#1baf7a", "#eda100"]  # slot1 blue, slot2 aqua, slot3 yellow

# Deterministic dialogue picks: the 3 longest Session5 (test) dialogues by
# utterance count, so the trajectory has enough timesteps to be legible --
# not cherry-picked by outcome (chosen before looking at any edge-norm
# values), just by length.
N_SAMPLE_DIALOGUES = 3


def pick_sample_dialogue_ids(n: int) -> list[int]:
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    counts = df.groupby("dialogue_id").size().sort_values(ascending=False)
    return counts.index[:n].tolist()


def trace_edge_norms_for_dialogue(model: RapportModel, dataset: MELDCachedDataset, dialogue_idx: int) -> list[float]:
    item = dataset[dialogue_idx]
    batch = collate_dialogues([item])

    norms: list[float] = []
    original = RelationalEdgeMemory.update_incident_edges

    def instrumented(self, edge_state, s, others, u_e, h_s_prev, node_state):
        original(self, edge_state, s, others, u_e, h_s_prev, node_state)
        if others:
            key = tuple(sorted((s, others[0])))
            # pair_index is symmetric (dyadic -> exactly one key ever used).
            for v in edge_state.values():
                norms.append(v.norm().item())
                break

    RelationalEdgeMemory.update_incident_edges = instrumented
    try:
        with torch.no_grad():
            model(
                batch["video_feat"], batch["audio_feat"], batch["text_feat"],
                batch["speaker_ids"], batch["dialogue_mask"], text_logits=batch["text_logits"],
                video_tokens=batch["video_tokens"], video_tokens_mask=batch["video_tokens_mask"],
                audio_tokens=batch["audio_tokens"], audio_tokens_mask=batch["audio_tokens_mask"],
            )
    finally:
        RelationalEdgeMemory.update_incident_edges = original

    return norms


def main() -> None:
    device = torch.device("cpu")
    # full_R_iemocap's actual config (scripts/run_iemocap_residual_matrix.py's
    # CONFIGS): relational=shift=temporal=residual=True -- must match exactly
    # for load_state_dict to succeed (temporal adds the A/V attention-pool params).
    model = RapportModel(num_classes=NUM_CLASSES, relational=True, shift=True, temporal=True, residual=True).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = MELDCachedDataset(
        PROCESSED_DIR / "test.parquet", CACHE_DIR, text_cache_subdir="text_ctx_iemocap",
        load_text_logits=True, load_av_tokens=True,
    )
    # dataset.dialogues is index-aligned with __getitem__ positions, sorted by dialogue_id.
    all_dialogue_ids = [int(d["dialogue_id"].iloc[0]) for d in dataset.dialogues]

    sample_ids = pick_sample_dialogue_ids(N_SAMPLE_DIALOGUES)
    print(f"[iemocap_edge_state_trajectory] sample dialogue_ids (longest {N_SAMPLE_DIALOGUES} in Session5/test): {sample_ids}")

    traces = {}
    for did in sample_ids:
        idx = all_dialogue_ids.index(did)
        norms = trace_edge_norms_for_dialogue(model, dataset, idx)
        traces[did] = norms
        print(f"[iemocap_edge_state_trajectory] dialogue_id={did}: {len(norms)} edge updates, "
              f"norm range [{min(norms):.3f}, {max(norms):.3f}]")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (did, norms) in enumerate(traces.items()):
        ax.plot(range(1, len(norms) + 1), norms, color=COLORS[i % len(COLORS)], marker="o", markersize=4,
                linewidth=1.5, label=f"dialogue_id={did} (n={len(norms)} turns)")
    ax.set_xlabel("Edge update index (dialogue time, one per speaking turn)")
    ax.set_ylabel("Edge-state ||e_sj|| (L2 norm)")
    ax.set_title("IEMOCAP Full stack, seed 42: dyadic edge-state norm over dialogue time\n(3 longest Session 5/test dialogues)", fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig_path = DOCS_DIR / "iemocap_edge_state_trajectory.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[iemocap_edge_state_trajectory] wrote {fig_path}")

    (PROJECT_ROOT / "outputs" / "iemocap_edge_state_trajectory_data.json").write_text(
        json.dumps({str(did): norms for did, norms in traces.items()}, indent=2)
    )


if __name__ == "__main__":
    main()
