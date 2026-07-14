import torch

from rapport.eval.rank_diagnostics import (
    posthoc_adjustment_sweep,
    predicted_vs_true_histogram,
    rank_of_true_class_stats,
    top1_confusion_for_classes,
)

CLASSES = ["a", "b", "c"]


def test_predicted_vs_true_histogram():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
    labels = torch.tensor([0, 0, 1])
    out = predicted_vs_true_histogram(logits, labels, CLASSES)
    assert out["predicted"] == {"a": 1, "b": 2, "c": 0}
    assert out["true"] == {"a": 2, "b": 1, "c": 0}


def test_rank_of_true_class_stats():
    # example 0: true=0, logits sorted desc are [0,1,2] -> true class rank 1
    # example 1: true=2, logits sorted desc are [0,1,2] -> true class rank 3 (worst)
    logits = torch.tensor([[5.0, 3.0, 1.0], [5.0, 3.0, 1.0]])
    labels = torch.tensor([0, 2])
    stats = rank_of_true_class_stats(logits, labels, CLASSES)
    assert stats["a"]["mean_rank_of_true_class"] == 1.0
    assert stats["a"]["rank_distribution"] == {1: 1, 2: 0, 3: 0}
    assert stats["c"]["mean_rank_of_true_class"] == 3.0
    assert stats["c"]["rank_distribution"] == {1: 0, 2: 0, 3: 1}
    assert "b" not in stats  # no true-b examples in this batch


def test_top1_confusion_for_classes():
    logits = torch.tensor(
        [
            [0.0, 5.0, 0.0],  # predicted b
            [0.0, 0.0, 5.0],  # predicted c
            [0.0, 5.0, 0.0],  # predicted b
        ]
    )
    labels = torch.tensor([0, 0, 0])  # all true=a
    out = top1_confusion_for_classes(logits, labels, CLASSES, target_classes=["a"])
    assert out["a"]["support"] == 3
    assert out["a"]["predicted_distribution"] == {"a": 0, "b": 2, "c": 1}


def test_posthoc_adjustment_sweep_boosts_rare_class():
    # class "c" is rare (prior 0.1); with tau=0 the model always predicts "a"
    # for a true-"c" example whose raw logits barely favor "a" over "c" --
    # a large enough tau should flip that prediction to "c".
    logits = torch.tensor([[1.0, 0.0, 0.9]])
    labels = torch.tensor([2])
    log_priors = torch.log(torch.tensor([0.8, 0.1, 0.1]))

    rows = posthoc_adjustment_sweep(logits, labels, log_priors, [0.0, 2.0], CLASSES)
    # tau=0: no adjustment, prediction stays "a", true class "c" gets 0 F1
    assert rows[0]["per_class_f1"]["c"] == 0.0
    # tau=2: adjustment large enough to flip the prediction to the true class "c"
    assert rows[1]["per_class_f1"]["c"] == 1.0
