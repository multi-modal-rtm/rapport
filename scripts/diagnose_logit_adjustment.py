"""Phase T gate-failure diagnosis: why did trained-time logit adjustment
fail to rescue fear/disgust on the seed-42 checkpoint? Diagnostics only --
loads the existing outputs/context_text_seed42/best_model.pt, no retraining.

STEP 1 (numerical LA audit) is done inline in the investigation session, not
here -- see docs/PHASE_T_DIAGNOSIS.md.

STEP 2: prediction anatomy on the test split -- predicted-class histogram,
and for every true-fear/true-disgust test utterance, the rank of the
correct class within that example's logit ordering (rank 1 = top logit).

STEP 3: post-hoc (inference-time-only) logit adjustment sweep on VAL for
tau_eval in {0.25, 0.5, 0.75, 1.0}: logits - tau_eval * log(prior), report
weighted F1 / macro F1 / fear F1 / disgust F1 per tau_eval. If a candidate
clears (all 7 nonzero AND weighted F1 >= 0.59), also applies that tau_eval
to TEST once and reports it.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rapport.data.constants import EMOTION_LABELS
from rapport.data.context_text import ContextTextCollator, MELDContextTextDataset
from rapport.eval.rank_diagnostics import posthoc_adjustment_sweep, rank_of_true_class_stats
from rapport.models.text_classifier import ContextTextClassifier
from rapport.training.losses import compute_class_priors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = len(EMOTION_LABELS)
CHECKPOINT = PROJECT_ROOT / "outputs" / "context_text_seed42" / "best_model.pt"


def load_model_and_priors(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = ContextTextClassifier(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_df = pd.read_parquet(PROJECT_ROOT / "data" / "meld" / "processed" / "train.parquet")
    priors = compute_class_priors(train_df["label"], NUM_CLASSES).to(device)
    return model, tokenizer, priors, ckpt


def build_loader(split: str, tokenizer, k: int = 8, max_length: int = 256) -> DataLoader:
    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    dataset = MELDContextTextDataset(processed_dir / f"{split}.parquet", tokenizer, k=k, max_length=max_length)
    collate = ContextTextCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def step2_prediction_anatomy(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = logits.argmax(dim=-1)
    pred_hist = {EMOTION_LABELS[c]: int((preds == c).sum()) for c in range(NUM_CLASSES)}
    true_hist = {EMOTION_LABELS[c]: int((labels == c).sum()) for c in range(NUM_CLASSES)}

    return {
        "predicted_class_histogram": pred_hist,
        "true_class_histogram": true_hist,
        "per_class_rank_of_true_stats": rank_of_true_class_stats(logits, labels, EMOTION_LABELS),
    }


def step3_posthoc_sweep(logits: torch.Tensor, labels: torch.Tensor, log_priors: torch.Tensor, taus: list[float]) -> list[dict]:
    rows = posthoc_adjustment_sweep(logits, labels.cpu(), log_priors.cpu(), taus, EMOTION_LABELS)
    for row in rows:
        row["all_7_nonzero"] = row.pop("all_classes_nonzero")  # preserve this script's original key name
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, priors, ckpt = load_model_and_priors(device)
    log_priors = torch.log(priors)

    print(f"[info] loaded checkpoint from epoch {ckpt['epoch']}, val_macro_f1={ckpt['val_macro_f1']:.4f}")

    dev_loader = build_loader("dev", tokenizer)
    test_loader = build_loader("test", tokenizer)

    dev_logits, dev_labels = collect_logits(model, dev_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    # sanity: raw-logit argmax on dev must reproduce the training run's logged val metrics
    dev_preds = dev_logits.argmax(dim=-1).tolist()
    sanity_wf1 = f1_score(dev_labels.tolist(), dev_preds, average="weighted", zero_division=0)
    sanity_mf1 = f1_score(dev_labels.tolist(), dev_preds, average="macro", zero_division=0)
    print(f"[sanity] recomputed val weighted_f1={sanity_wf1:.4f} macro_f1={sanity_mf1:.4f} (expect 0.5780 / 0.3800)")

    step2 = step2_prediction_anatomy(test_logits, test_labels)
    print("\n[step 2] predicted-class histogram (test):", step2["predicted_class_histogram"])
    print("[step 2] true-class histogram (test):", step2["true_class_histogram"])
    for name in ("fear", "disgust"):
        stats = step2["per_class_rank_of_true_stats"][name]
        print(f"[step 2] {name}: support={stats['support']} mean_rank={stats['mean_rank_of_true_class']:.2f} "
              f"median_rank={stats['median_rank_of_true_class']:.1f} rank_dist={stats['rank_distribution']}")

    taus = [0.25, 0.5, 0.75, 1.0]
    step3_val = step3_posthoc_sweep(dev_logits, dev_labels, log_priors, taus)
    print("\n[step 3] VAL post-hoc sweep:")
    for row in step3_val:
        print(f"  tau_eval={row['tau_eval']:.2f} weighted_f1={row['weighted_f1']:.4f} macro_f1={row['macro_f1']:.4f} "
              f"fear_f1={row['per_class_f1']['fear']:.3f} disgust_f1={row['per_class_f1']['disgust']:.3f} "
              f"all_7_nonzero={row['all_7_nonzero']}")

    candidates = [r for r in step3_val if r["all_7_nonzero"] and r["weighted_f1"] >= 0.59]
    step3_test = None
    chosen_tau = None
    if candidates:
        # highest val macro F1 among qualifying candidates, consistent with this
        # project's prior tau-selection convention (docs/RECIPE.md)
        best = max(candidates, key=lambda r: r["macro_f1"])
        chosen_tau = best["tau_eval"]
        step3_test = step3_posthoc_sweep(test_logits, test_labels, log_priors, [chosen_tau])[0]
        print(f"\n[step 3] candidate found: tau_eval={chosen_tau} -- applying once to TEST")
        print(f"  TEST weighted_f1={step3_test['weighted_f1']:.4f} macro_f1={step3_test['macro_f1']:.4f} "
              f"per_class_f1={step3_test['per_class_f1']} all_7_nonzero={step3_test['all_7_nonzero']}")
    else:
        print("\n[step 3] no tau_eval on VAL satisfies (all_7_nonzero AND weighted_f1 >= 0.59)")

    out = {
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_macro_f1": ckpt["val_macro_f1"],
        "sanity_recomputed_val_weighted_f1": sanity_wf1,
        "sanity_recomputed_val_macro_f1": sanity_mf1,
        "log_priors": {label: float(lp) for label, lp in zip(EMOTION_LABELS, log_priors.tolist())},
        "step2_prediction_anatomy": step2,
        "step3_val_sweep": step3_val,
        "step3_chosen_tau_eval": chosen_tau,
        "step3_test_result": step3_test,
    }
    out_path = PROJECT_ROOT / "outputs" / "logit_adjustment_diagnosis.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
