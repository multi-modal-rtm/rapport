import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_inverse_frequency_alpha(labels: pd.Series, num_classes: int) -> torch.Tensor:
    """Class-balanced alpha weights, normalized to mean 1: alpha_c = inv_freq_c *
    (num_classes / sum(inv_freq)), inv_freq_c = total / (num_classes * count_c).

    See docs/DIAGNOSIS.md, GATE-FAILURE INVESTIGATION: disgust/fear (MELD's two
    rarest classes) got exactly 0.0 val F1 across every epoch of every seed
    under gamma-only focal loss -- not a checkpoint-selection artifact (ruled
    out directly) or a loss-implementation bug (audited clean against Lin et
    al. 2017). Gamma alone reweights easy vs. hard examples but applies no
    class-frequency correction, which is exactly the gap this closes.
    """
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0)
    total = counts.sum()
    inv_freq = total / (num_classes * counts.clip(lower=1))
    alpha = inv_freq * (num_classes / inv_freq.sum())
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
