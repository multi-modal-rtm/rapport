"""PAPER-ASSETS PHASE, figure (e): confusion matrices, both corpora,
matched color scales. Inference-only (loads an already-trained checkpoint,
forward pass on the test split, no gradient updates -- not a training
run): full_R_seed42 (MELD) and full_R_iemocap_seed42 (IEMOCAP), the two
headline "graph stack" configs. Row-normalized (recall within each true
class) so the shared [0,1] color scale is meaningful despite MELD/IEMOCAP
having different class counts and test-set sizes.

Usage:
    uv run python -m scripts.generate_confusion_matrices
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.data.constants import EMOTION_LABELS as MELD_LABELS
from rapport.data.constants import IEMOCAP_EMOTION_LABELS as IEMOCAP_LABELS
from rapport.models.rapport_model import RapportModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "paper_assets" / "figures"


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def compute_confusion(
    checkpoint_path: Path, num_classes: int, processed_dir: Path, cache_dir: Path,
    text_cache_subdir: str, test_filename: str,
) -> tuple[list, list]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RapportModel(num_classes=num_classes, relational=True, shift=True, temporal=True, residual=True).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = MELDCachedDataset(
        processed_dir / test_filename, cache_dir, text_cache_subdir=text_cache_subdir,
        load_text_logits=True, load_av_tokens=True,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_dialogues, num_workers=0)

    all_preds, all_labels = [], []
    for batch in loader:
        batch = _move(batch, device)
        logits, _ = model(
            batch["video_feat"], batch["audio_feat"], batch["text_feat"], batch["speaker_ids"], batch["dialogue_mask"],
            video_tokens=batch["video_tokens"], video_tokens_mask=batch["video_tokens_mask"],
            audio_tokens=batch["audio_tokens"], audio_tokens_mask=batch["audio_tokens_mask"],
            text_logits=batch["text_logits"],
        )
        mask = batch["dialogue_mask"]
        preds = logits[mask].argmax(dim=-1).cpu().tolist()
        labels = batch["labels"][mask].cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
    return all_labels, all_preds


def main() -> None:
    meld_labels, meld_preds = compute_confusion(
        OUTPUTS_DIR / "full_R_seed42" / "best_model.pt", len(MELD_LABELS),
        PROJECT_ROOT / "data" / "meld" / "processed", PROJECT_ROOT / "data" / "meld" / "cache",
        "text_ctx", "test.parquet",
    )
    iemocap_labels, iemocap_preds = compute_confusion(
        OUTPUTS_DIR / "full_R_iemocap_seed42" / "best_model.pt", len(IEMOCAP_LABELS),
        PROJECT_ROOT / "data" / "iemocap" / "processed", PROJECT_ROOT / "data" / "iemocap" / "cache",
        "text_ctx_iemocap", "test.parquet",
    )

    meld_cm = confusion_matrix(meld_labels, meld_preds, labels=list(range(len(MELD_LABELS))))
    iemocap_cm = confusion_matrix(iemocap_labels, iemocap_preds, labels=list(range(len(IEMOCAP_LABELS))))

    meld_cm_norm = meld_cm.astype(float) / meld_cm.sum(axis=1, keepdims=True)
    iemocap_cm_norm = iemocap_cm.astype(float) / iemocap_cm.sum(axis=1, keepdims=True)

    (OUTPUTS_DIR / "confusion_matrices_raw.json").write_text(
        json.dumps(
            {
                "meld_full_R_seed42": {"labels": MELD_LABELS, "matrix": meld_cm.tolist()},
                "iemocap_full_R_iemocap_seed42": {"labels": IEMOCAP_LABELS, "matrix": iemocap_cm.tolist()},
            },
            indent=2,
        )
    )

    # Explicit GridSpec (rather than fig.colorbar(ax=<list>), which steals
    # space from the existing axes via its own internal layout pass and
    # interacts unpredictably with bbox_inches="tight" -- that combination
    # is what previously cost the left panel its y-axis label under some
    # renders) so the colorbar gets its own dedicated axis, and both
    # confusion-matrix axes keep a fixed, equal amount of room for their
    # y-axis label and tick labels regardless of colorbar placement.
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], hspace=0.65, wspace=0.32)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[1, :])

    for ax, cm_norm, labels, title in (
        (axes[0], meld_cm_norm, MELD_LABELS, "MELD (Full stack, seed 42)"),
        (axes[1], iemocap_cm_norm, IEMOCAP_LABELS, "IEMOCAP (Full stack, seed 42)"),
    ):
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=13)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(
                    j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8,
                )

    # Single colorbar on its own dedicated axis spanning both columns,
    # centered below both panels -- the shared [0,1] scale reads as
    # serving both matrices rather than belonging to IEMOCAP alone.
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Row-normalized (recall within true class)")
    fig.suptitle("Confusion matrices, matched [0,1] color scale (recall-normalized)", fontsize=13)

    for ext in ("pdf", "png"):
        fig_path = FIGURES_DIR / f"confusion_matrices.{ext}"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"[generate_confusion_matrices] wrote {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
