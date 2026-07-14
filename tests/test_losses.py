import pandas as pd
import torch

from rapport.training.losses import FocalLoss, LogitAdjustedLoss, compute_class_priors, compute_tempered_alpha


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


def test_compute_class_priors_sums_to_one():
    labels = pd.Series([0] * 90 + [1] * 10)
    priors = compute_class_priors(labels, num_classes=2)
    assert torch.isclose(priors.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(priors[0], torch.tensor(0.9), atol=1e-5)
    assert torch.isclose(priors[1], torch.tensor(0.1), atol=1e-5)


def test_logit_adjustment_boosts_rare_class_logit():
    """Menon et al. 2021: subtracting tau*log(prior_c) *adds* a larger
    positive offset to rarer classes (more negative log-prior), so an
    identical raw logit for a common vs. rare class should come out higher,
    post-adjustment, for the rare class.
    """
    priors = torch.tensor([0.9, 0.1])
    loss_fn = LogitAdjustedLoss(priors, tau=1.0)

    logits = torch.tensor([[1.0, 1.0]])  # identical raw logits for both classes
    adjusted = logits - loss_fn.tau * loss_fn.log_priors
    assert adjusted[0, 1] > adjusted[0, 0], "the rare class's adjusted logit should be boosted above the common class's"


def test_logit_adjustment_tau_zero_is_plain_cross_entropy():
    torch.manual_seed(0)
    priors = torch.tensor([0.9, 0.1])
    logits = torch.randn(5, 2)
    targets = torch.tensor([0, 1, 0, 1, 0])

    loss_fn = LogitAdjustedLoss(priors, tau=0.0)
    import torch.nn.functional as F

    assert torch.isclose(loss_fn(logits, targets), F.cross_entropy(logits, targets))


def test_logit_adjustment_respects_ignore_index():
    priors = torch.tensor([0.5, 0.5])
    loss_fn = LogitAdjustedLoss(priors, tau=1.0, ignore_index=-1)

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [5.0, -5.0]])
    targets_with_pad = torch.tensor([0, 1, -1])
    targets_without_pad = torch.tensor([0, 1])

    loss_with_pad = loss_fn(logits, targets_with_pad)
    loss_without_pad = loss_fn(logits[:2], targets_without_pad)
    assert torch.isclose(loss_with_pad, loss_without_pad)


def test_focal_loss_without_alpha_is_unchanged():
    """Backward compatibility: omitting alpha must reproduce the original gamma-only loss."""
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, -1])

    fl_no_alpha = FocalLoss(gamma=3.0, ignore_index=-1)
    fl_alpha_ones = FocalLoss(gamma=3.0, ignore_index=-1, alpha=torch.ones(3))

    assert torch.allclose(fl_no_alpha(logits, targets), fl_alpha_ones(logits, targets))
