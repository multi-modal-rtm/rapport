import torch

from rapport.models.temporal_attention import TemporalAttentionPool


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def test_init_output_matches_masked_mean_no_padding():
    torch.manual_seed(0)
    pool = TemporalAttentionPool(dim=64, num_heads=4)
    tokens = torch.randn(5, 10, 64)
    mask = torch.ones(5, 10, dtype=torch.bool)

    out = pool(tokens, mask)
    expected = _masked_mean(tokens, mask)

    assert torch.allclose(out, expected, atol=1e-5)


def test_init_output_matches_masked_mean_with_padding():
    torch.manual_seed(1)
    pool = TemporalAttentionPool(dim=64, num_heads=4)
    tokens = torch.randn(4, 12, 64)
    mask = torch.zeros(4, 12, dtype=torch.bool)
    lengths = [12, 7, 1, 5]
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    out = pool(tokens, mask)
    expected = _masked_mean(tokens, mask)

    assert torch.allclose(out, expected, atol=1e-5)


def test_padded_positions_do_not_influence_output():
    torch.manual_seed(2)
    pool = TemporalAttentionPool(dim=32, num_heads=4)
    tokens = torch.randn(3, 8, 32)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[:, :5] = True

    out_before = pool(tokens, mask)

    # Corrupt only the padded positions with huge noise -- output must be unchanged.
    tokens_corrupted = tokens.clone()
    tokens_corrupted[:, 5:] = 1e6 * torch.randn(3, 3, 32)
    out_after = pool(tokens_corrupted, mask)

    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_padded_positions_do_not_influence_output_after_training():
    """The masking discipline must hold generically, not just at init --
    train the module briefly (breaking the zero-init symmetry) and re-check.
    """
    torch.manual_seed(3)
    pool = TemporalAttentionPool(dim=32, num_heads=4)
    optimizer = torch.optim.AdamW(pool.parameters(), lr=1e-2)

    tokens = torch.randn(6, 10, 32)
    mask = torch.zeros(6, 10, dtype=torch.bool)
    mask[:, :6] = True
    target = torch.randn(6, 32)

    for _ in range(20):
        optimizer.zero_grad()
        out = pool(tokens, mask)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    out_before = pool(tokens, mask)
    tokens_corrupted = tokens.clone()
    tokens_corrupted[:, 6:] = 1e6 * torch.randn(6, 4, 32)
    out_after = pool(tokens_corrupted, mask)

    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_gradient_flows_to_all_parameters():
    torch.manual_seed(4)
    pool = TemporalAttentionPool(dim=32, num_heads=4)
    tokens = torch.randn(3, 6, 32)
    mask = torch.ones(3, 6, dtype=torch.bool)

    out = pool(tokens, mask)
    out.sum().backward()

    for name, param in pool.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient at {name}"


def test_pool_is_trainable():
    torch.manual_seed(5)
    pool = TemporalAttentionPool(dim=16, num_heads=4)
    optimizer = torch.optim.AdamW(pool.parameters(), lr=1e-2)

    tokens = torch.randn(4, 8, 16)
    mask = torch.ones(4, 8, dtype=torch.bool)
    target = torch.randn(4, 16)

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        out = pool(tokens, mask)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1
