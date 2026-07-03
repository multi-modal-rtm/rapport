import pandas as pd
import torch

from rapport.training.losses import FocalLoss, compute_tempered_alpha


def test_compute_tempered_alpha_favors_rare_classes():
    # class 0: 100 examples (common), class 1: 10 examples (rare)
    labels = pd.Series([0] * 100 + [1] * 10)
    alpha = compute_tempered_alpha(labels, num_classes=2, tau=1.0)

    assert alpha[1] > alpha[0], "the rarer class must get a larger alpha weight"
    assert torch.isclose(alpha.mean(), torch.tensor(1.0), atol=1e-5), "alpha should be normalized to mean 1"


def test_compute_tempered_alpha_tau_zero_is_uniform():
    labels = pd.Series([0] * 100 + [1] * 10)
    alpha = compute_tempered_alpha(labels, num_classes=2, tau=0.0)
    assert torch.allclose(alpha, torch.ones(2)), "tau=0 must reproduce the gamma-only (no class correction) case"


def test_compute_tempered_alpha_interpolates_monotonically():
    labels = pd.Series([0] * 100 + [1] * 10)
    ratios = [
        (compute_tempered_alpha(labels, num_classes=2, tau=tau)[1]
         / compute_tempered_alpha(labels, num_classes=2, tau=tau)[0]).item()
        for tau in (0.0, 0.25, 0.5, 1.0)
    ]
    assert ratios == sorted(ratios), "rare-class:common-class alpha ratio must increase monotonically with tau"


def test_focal_loss_alpha_upweights_rare_class_errors():
    """A wrong prediction on the rare class should contribute more to the loss
    than an equally-wrong prediction on the common class, once alpha is set --
    this is the entire point of the amendment (docs/DIAGNOSIS.md).
    """
    torch.manual_seed(0)
    alpha = torch.tensor([0.2, 1.8])  # common class 0 downweighted, rare class 1 upweighted

    logits_wrong_on_common = torch.tensor([[0.0, 5.0]])  # predicts class 1
    logits_wrong_on_rare = torch.tensor([[5.0, 0.0]])  # predicts class 0

    fl = FocalLoss(gamma=3.0, ignore_index=-1, alpha=alpha)
    loss_common_wrong = fl(logits_wrong_on_common, torch.tensor([0]))  # true=0, predicted 1: common-class error
    loss_rare_wrong = fl(logits_wrong_on_rare, torch.tensor([1]))  # true=1, predicted 0: rare-class error

    assert loss_rare_wrong > loss_common_wrong, (
        "a wrong prediction on the rare (higher-alpha) true class should be penalized more"
    )


def test_focal_loss_without_alpha_is_unchanged():
    """Backward compatibility: omitting alpha must reproduce the original gamma-only loss."""
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, -1])

    fl_no_alpha = FocalLoss(gamma=3.0, ignore_index=-1)
    fl_alpha_ones = FocalLoss(gamma=3.0, ignore_index=-1, alpha=torch.ones(3))

    assert torch.allclose(fl_no_alpha(logits, targets), fl_alpha_ones(logits, targets))
