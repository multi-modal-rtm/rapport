"""Rank/confusion/post-hoc-adjustment diagnostics for a trained classifier's
raw logits, used to distinguish "signal exists but the decision threshold
is off" (post-hoc logit adjustment should fix it) from "the representation
never learned this class" (no adjustment variant can fix it) -- see
docs/PHASE_T_DIAGNOSIS.md and docs/PHASE_T_STEP4.md.

Shared between scripts/diagnose_logit_adjustment.py (Phase T Steps 2-3) and
scripts/train_context_text.py (Step 4's post-training instrumentation), so
both use one audited implementation instead of two copies.
"""

from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def predicted_vs_true_histogram(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict:
    preds = logits.argmax(dim=-1)
    num_classes = len(class_names)
    return {
        "predicted": {class_names[c]: int((preds == c).sum()) for c in range(num_classes)},
        "true": {class_names[c]: int((labels == c).sum()) for c in range(num_classes)},
    }


def rank_of_true_class_stats(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict:
    """For every example, the rank of its TRUE class within that example's own
    logit ordering (1 = highest-scoring class), aggregated per true class.
    """
    num_classes = len(class_names)
    order = logits.argsort(dim=-1, descending=True)  # [N, C], class ids sorted by score
    rank_of_true = (order == labels.unsqueeze(1)).float().argmax(dim=1) + 1  # [N], 1-indexed

    stats = {}
    for c, name in enumerate(class_names):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ranks = rank_of_true[mask]
        stats[name] = {
            "support": int(mask.sum()),
            "mean_rank_of_true_class": float(ranks.float().mean()),
            "median_rank_of_true_class": float(ranks.float().median()),
            "rank_distribution": {int(r): int((ranks == r).sum()) for r in range(1, num_classes + 1)},
        }
    return stats


def top1_confusion_for_classes(
    logits: torch.Tensor, labels: torch.Tensor, class_names: list[str], target_classes: list[str]
) -> dict:
    """For each class in target_classes, the distribution of top-1 PREDICTED
    classes (raw logits, no adjustment) among utterances whose TRUE label is
    that class. Semantically adjacent errors (e.g. fear -> surprise,
    disgust -> anger) indicate coarse-but-present signal; errors scattering
    toward the majority classes (neutral/joy) indicate no signal.
    """
    preds = logits.argmax(dim=-1)
    num_classes = len(class_names)
    out = {}
    for name in target_classes:
        c = class_names.index(name)
        mask = labels == c
        if mask.sum() == 0:
            continue
        class_preds = preds[mask]
        out[name] = {
            "support": int(mask.sum()),
            "predicted_distribution": {class_names[p]: int((class_preds == p).sum()) for p in range(num_classes)},
        }
    return out


def posthoc_adjustment_sweep(
    logits: torch.Tensor, labels: torch.Tensor, log_priors: torch.Tensor, taus: list[float], class_names: list[str]
) -> list[dict]:
    """logits - tau*log_priors at each tau in taus (inference-time only --
    does not modify the model or retrain anything)."""
    num_classes = len(class_names)
    labels_list = labels.tolist()
    rows = []
    for tau in taus:
        adjusted = logits - tau * log_priors.to(logits.device)
        preds = adjusted.argmax(dim=-1).tolist()
        per_class_f1 = f1_score(labels_list, preds, average=None, labels=list(range(num_classes)), zero_division=0)
        rows.append(
            {
                "tau_eval": tau,
                "weighted_f1": f1_score(labels_list, preds, average="weighted", zero_division=0),
                "macro_f1": f1_score(labels_list, preds, average="macro", zero_division=0),
                "per_class_f1": {label: float(f1) for label, f1 in zip(class_names, per_class_f1)},
                "all_classes_nonzero": bool(all(f1 > 0 for f1 in per_class_f1)),
            }
        )
    return rows
