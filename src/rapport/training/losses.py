import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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
