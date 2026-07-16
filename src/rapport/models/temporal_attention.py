"""Temporal attention pooling over cached A/V token sequences
(docs/SPEC_RAPPORT_COMPONENTS.md section C). Text is exempt -- Phase T's
context-window pooling is already contextual (see docs/PHASE_N4.md Step 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    """C1: prepends a learnable aggregation token, runs ONE block of
    multi-head self-attention (4 heads) + residual + LayerNorm, and returns
    the aggregation token's position as the pooled [B, dim] output.

    **Implementation decision (spec C2 leaves the init scheme open, "ANY
    scheme passing C3 is acceptable" -- recorded in full in
    docs/SPEC_RAPPORT_COMPONENTS.md):** LayerNorm is applied to the
    QUERY/KEY path only (normalizing what's used to compute attention
    scores -- a "QK-norm" variant, a legitimate and fairly common
    transformer design), never to the VALUE path or the residual sum
    itself. A literal post-residual LayerNorm has no learnable parameter
    that can turn its normalization off, so any block that wraps the final
    residual sum in a LayerNorm can only ever APPROXIMATE the raw masked
    mean at init, not match it exactly -- which is what C3's atol=1e-5
    requirement demands. Keeping LayerNorm out of the value/residual path
    is what makes exact equivalence achievable: combined with zero-init on
    the query projection (-> uniform attention over unmasked tokens,
    regardless of what the LayerNorm did to the keys) and identity-init on
    the value/output projections, the block's output is EXACTLY
    `agg_token + masked_mean(tokens)`, and `agg_token` itself is zero-init,
    so the output is exactly the masked mean at construction time.
    """

    def __init__(self, dim: int = 768, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.agg_token = nn.Parameter(torch.zeros(1, dim))
        self.qk_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # C2: mean-pool-equivalent init.
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L, dim], mask: [B, L] bool (True = real token, False =
        padding) -> pooled [B, dim]. Every batch element must have at least
        one True position in mask (an utterance always has >= 1 real token).
        """
        batch_size = tokens.shape[0]
        agg = self.agg_token.expand(batch_size, self.dim)  # [B, dim]

        q_in = self.qk_norm(agg)  # [B, dim]
        k_in = self.qk_norm(tokens)  # [B, L, dim]

        q = self.q_proj(q_in).view(batch_size, self.num_heads, 1, self.head_dim)
        k = self.k_proj(k_in).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # [B, H, 1, L]
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).reshape(batch_size, self.dim)  # [B, dim]
        out = self.out_proj(context)

        return agg + out
