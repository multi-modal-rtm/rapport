"""Phase N4-R Step 1: diagnostics from EXISTING Phase N4 ablation-matrix
artifacts (checkpoints + metrics.json) -- no training, forward-pass eval
only.

  1a. Train-vs-val gap per config/seed (overfitting fingerprint): loads
      each of the 15 best_model.pt checkpoints and evaluates them on the
      TRAIN split (forward pass only), compared against that same
      checkpoint's already-logged val metrics.
  1b. Learning curves: did `full` ever exceed `base_fusion`'s peak val
      metric at any epoch, per seed, or was it dominated throughout?
  1d. base_fusion's learned fusion weights: Frobenius norm of W_proj's
      text_ctx / audio / video column blocks, per seed.

(1c restates docs/SHIFT_LABEL_STATS.md's numbers -- no computation needed,
handled directly in the write-up doc.)

Usage:
    uv run python -m scripts.diagnose_n4_gate_failure
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rapport.data.constants import EMOTION_LABELS
from rapport.models.rapport_model import RapportModel
from scripts.train_rapport import build_dataloaders, evaluate_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = len(EMOTION_LABELS)

CONFIGS: dict[str, dict[str, bool]] = {
    "full": {"relational": True, "shift": True, "temporal": True},
    "minus_relational": {"relational": False, "shift": True, "temporal": True},
    "minus_shift": {"relational": True, "shift": False, "temporal": True},
    "minus_temporal": {"relational": True, "shift": True, "temporal": False},
    "base_fusion": {"relational": False, "shift": False, "temporal": False},
}
SEEDS = (42, 1337, 2024)


def load_checkpoint(config_name: str, seed: int, device: torch.device) -> tuple[RapportModel, dict]:
    flags = CONFIGS[config_name]
    run_dir = PROJECT_ROOT / "outputs" / f"{config_name}_seed{seed}"
    model = RapportModel(num_classes=NUM_CLASSES, **flags).to(device)
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return model, metrics


def step_1a_train_val_gap(device: torch.device) -> dict:
    print("\n=== STEP 1a: train-vs-val gap per config/seed ===")
    results: dict[str, dict] = {}
    loaders_by_temporal = {
        temporal: build_dataloaders(temporal=temporal) for temporal in (True, False)
    }

    for config_name, flags in CONFIGS.items():
        loaders = loaders_by_temporal[flags["temporal"]]
        for seed in SEEDS:
            run_name = f"{config_name}_seed{seed}"
            model, metrics = load_checkpoint(config_name, seed, device)
            train_w, train_m, _, _, _, _, _ = evaluate_split(model, loaders["train"], device)

            val_w = metrics["history"][metrics["best_epoch"]]["val_weighted_f1"]
            val_m = metrics["best_val_macro_f1"]

            row = {
                "train_weighted_f1": train_w,
                "train_macro_f1": train_m,
                "val_weighted_f1": val_w,
                "val_macro_f1": val_m,
                "gap_weighted_f1": train_w - val_w,
                "gap_macro_f1": train_m - val_m,
            }
            results[run_name] = row
            print(
                f"  {run_name:30s} train_w={train_w:.4f} val_w={val_w:.4f} gap_w={row['gap_weighted_f1']:+.4f} | "
                f"train_m={train_m:.4f} val_m={val_m:.4f} gap_m={row['gap_macro_f1']:+.4f}"
            )

    print("\n  mean gap_weighted_f1 by component count:")
    for config_name in CONFIGS:
        gaps = [results[f"{config_name}_seed{s}"]["gap_weighted_f1"] for s in SEEDS]
        n_components = sum(CONFIGS[config_name].values())
        print(f"    {config_name:20s} (n_components={n_components}) mean_gap_w={sum(gaps) / 3:.4f}")

    return results


def step_1b_learning_curves() -> dict:
    print("\n=== STEP 1b: did `full` ever exceed base_fusion's peak val metric? ===")
    results: dict[str, dict] = {}
    for seed in SEEDS:
        full_metrics = json.loads((PROJECT_ROOT / "outputs" / f"full_seed{seed}" / "metrics.json").read_text())
        base_metrics = json.loads((PROJECT_ROOT / "outputs" / f"base_fusion_seed{seed}" / "metrics.json").read_text())

        full_curve = [(h["epoch"], h["val_weighted_f1"], h["val_macro_f1"]) for h in full_metrics["history"]]
        base_peak_w = max(h["val_weighted_f1"] for h in base_metrics["history"])
        base_peak_m = max(h["val_macro_f1"] for h in base_metrics["history"])
        full_peak_w = max(h["val_weighted_f1"] for h in full_metrics["history"])
        full_peak_m = max(h["val_macro_f1"] for h in full_metrics["history"])

        ever_exceeded_w = full_peak_w > base_peak_w
        ever_exceeded_m = full_peak_m > base_peak_m

        results[f"seed{seed}"] = {
            "full_peak_weighted_f1": full_peak_w,
            "base_fusion_peak_weighted_f1": base_peak_w,
            "full_ever_exceeded_base_fusion_weighted": ever_exceeded_w,
            "full_peak_macro_f1": full_peak_m,
            "base_fusion_peak_macro_f1": base_peak_m,
            "full_ever_exceeded_base_fusion_macro": ever_exceeded_m,
            "full_curve": full_curve,
        }
        print(
            f"  seed{seed}: full_peak_w={full_peak_w:.4f} vs base_fusion_peak_w={base_peak_w:.4f} "
            f"-> ever_exceeded={ever_exceeded_w} | full_peak_m={full_peak_m:.4f} vs "
            f"base_fusion_peak_m={base_peak_m:.4f} -> ever_exceeded={ever_exceeded_m}"
        )
    return results


def step_1d_fusion_weight_norms(device: torch.device) -> dict:
    print("\n=== STEP 1d: base_fusion W_proj column-block norms (text_ctx / audio / video) ===")
    results: dict[str, dict] = {}
    for seed in SEEDS:
        model, _ = load_checkpoint("base_fusion", seed, device)
        w = model.fusion[0].weight.detach().cpu()  # [FUSION_DIM=256, 3*768=2304]
        text_block = w[:, 0:768]
        audio_block = w[:, 768:1536]
        video_block = w[:, 1536:2304]

        text_norm = text_block.norm().item()
        audio_norm = audio_block.norm().item()
        video_norm = video_block.norm().item()
        total = text_norm + audio_norm + video_norm

        results[f"seed{seed}"] = {
            "text_ctx_norm": text_norm,
            "audio_norm": audio_norm,
            "video_norm": video_norm,
            "text_ctx_frac": text_norm / total,
            "audio_frac": audio_norm / total,
            "video_frac": video_norm / total,
        }
        print(
            f"  seed{seed}: text_ctx={text_norm:.3f} ({text_norm / total:.1%}) "
            f"audio={audio_norm:.3f} ({audio_norm / total:.1%}) "
            f"video={video_norm:.3f} ({video_norm / total:.1%})"
        )
    return results


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {
        "step_1a_train_val_gap": step_1a_train_val_gap(device),
        "step_1b_learning_curves": step_1b_learning_curves(),
        "step_1d_fusion_weight_norms": step_1d_fusion_weight_norms(device),
    }

    out_path = PROJECT_ROOT / "outputs" / "n4r_step1_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
