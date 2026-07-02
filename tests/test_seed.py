import torch

from socialarcnet.seed import set_seed


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.randn(4, 4)

    set_seed(42)
    b = torch.randn(4, 4)

    assert torch.equal(a, b)
