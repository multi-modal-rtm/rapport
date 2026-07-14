import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_priors(labels: pd.Series, num_classes: int) -> torch.Tensor:
    """Per-class prior probability prior_c = count_c / total, from the train
    split's label distribution -- the prior term used by logit adjustment
    (Menon et al. 2021, "Long-Tail Learning via Logit Adjustment").
    `clip(lower=1)` avoids log(0) if a class is entirely absent from labels.
    """
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0)
    total = counts.sum()
    priors = counts.clip(lower=1) / total
    return torch.tensor(priors.to_numpy(), dtype=torch.float32)


class LogitAdjustedLoss(nn.Module):
    """Logit adjustment (Menon et al. 2021): subtract `tau * log(prior_c)`
    from the logits before standard cross-entropy. Rarer classes have more
    negative `log(prior_c)`, so this *adds* a larger positive offset to
    their logits during training, pushing the model to widen their margin
    to compensate -- unlike FocalLoss's alpha weighting (RECIPE.md), this is
    an additive logit-space correction derived directly from label
    frequency, not a multiplicative loss-weight.

    This must be used for the TRAINING loss only. At eval time, raw
    (unadjusted) logits are used directly for argmax/scoring -- callers must
    not wrap eval-time prediction in this module.
    """

    def __init__(self, priors: torch.Tensor, tau: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.tau = tau
        self.ignore_index = ignore_index
        self.register_buffer("log_priors", torch.log(priors))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]

        adjusted_logits = logits - self.tau * self.log_priors
        return F.cross_entropy(adjusted_logits, targets)


def compute_tempered_alpha(labels: pd.Series, num_classes: int, tau: float) -> torch.Tensor:
    """Tempered class-balanced alpha weights: w_c = (1/f_c)^tau, f_c = class c's
    train-split frequency, normalized to mean 1 across classes.

    tau is the single dial between the two failure modes documented in
    docs/DIAGNOSIS.md / docs/RECIPE.md:
    - tau=0: w_c = 1 for all c (gamma-only focal loss, no class correction) --
      disgust F1 was exactly 0.0 at every epoch of every seed under this.
    - tau=1: full inverse-frequency (the original amendment) -- fixes
      disgust/fear but overcorrects, dragging neutral recall (and therefore
      weighted F1 and raw accuracy) down hard, since neutral is 48% of test.
    tau in (0, 1) tempers the correction. Selected once on val (seed 42 only,
    fixed selection criterion), then locked -- see docs/RECIPE.md.
    """
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0)
    total = counts.sum()
    freq = counts.clip(lower=1) / total
    weights = freq ** (-tau)
    alpha = weights * (num_classes / weights.sum())
    return torch.tensor(alpha.to_numpy(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss with an ignore_index for padded dialogue positions.

    `alpha`, if given, is a per-class weight tensor [num_classes] (typically
    from compute_inverse_frequency_alpha) applied as the standard
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t). Absent alpha (the default),
    this is the gamma-only variant (alpha_t = 1), i.e. unchanged from before.
    """

    def __init__(self, gamma: float = 3.0, ignore_index: int = -1, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]

        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()
