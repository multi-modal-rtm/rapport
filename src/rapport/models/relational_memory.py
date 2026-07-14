"""Relational edge memory (docs/SPEC_RAPPORT_COMPONENTS.md, section A).

Maintains one 128-d edge state per UNORDERED speaker pair in a dialogue,
updated only for edges incident to the speaker currently uttering, and
read back both as an edge-conditioned GATv2-style attention context (feeds
the node GRU update) and as a plain mean (feeds the classification
readout alongside the node state). Edge states are stored in a dict keyed
by `pair_index(i, j)` rather than a dense `[P, 128]` tensor, since P (the
number of speakers who have appeared so far in a given dialogue) is only
known incrementally as the dialogue unrolls -- conceptually this dict IS
`E_t`, just with untouched pairs implicitly zero rather than materialized.

Ordering is fixed (spec A4) and implemented by the two-call contract
below: `update_incident_edges` (reads node states at t-1) must be called
BEFORE `attend` and `edge_mean` (which read the edge states just written,
i.e. "at t").
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pair_index(i: int, j: int) -> int:
    """Canonical flat index for the unordered pair {i, j}, i != j, i,j >= 0.

    Standard triangular numbering: for i < j, index = j*(j-1)/2 + i. This is
    a bijection from {(i,j) : 0 <= i < j} onto {0, 1, 2, ...} -- for a fixed
    max index n, the pairs with j <= n fill exactly [0, n*(n+1)/2) with no
    gaps or collisions (tested in tests/test_relational_memory.py).
    """
    if i == j:
        raise ValueError(f"pair_index requires i != j, got i=j={i}")
    if i > j:
        i, j = j, i
    return j * (j - 1) // 2 + i


class RelationalEdgeMemory(nn.Module):
    """One shared GRUCell for edge updates + one GATv2-style attention head
    for edge-conditioned message passing, per docs/SPEC_RAPPORT_COMPONENTS.md
    section A. Operates on a single (node_state, edge_state) dict pair
    belonging to one dialogue at a time -- callers loop over active
    (batch_element, speaker) pairs, same pattern as the base (non-relational)
    node update.
    """

    EDGE_DIM = 128

    def __init__(self, node_dim: int, utt_dim: int):
        super().__init__()
        self.node_dim = node_dim
        # A3: e_sj = GRUCell([U_e || h_s || h_j], e_prev) -- ONE shared cell.
        self.edge_gru = nn.GRUCell(utt_dim + 2 * node_dim, self.EDGE_DIM)
        # A5: alpha_sj ~ a^T LeakyReLU(W_n h_j + W_n h_s + W_e e_sj) -- ONE
        # shared W_n applied to both h_j and h_s (GATv2-style shared transform).
        self.W_n = nn.Linear(node_dim, self.EDGE_DIM, bias=False)
        self.W_e = nn.Linear(self.EDGE_DIM, self.EDGE_DIM, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(self.EDGE_DIM))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        # A5: m_sj = W_m[h_j || e_sj]
        self.W_m = nn.Linear(node_dim + self.EDGE_DIM, self.EDGE_DIM)

    def update_incident_edges(
        self,
        edge_state: dict[int, torch.Tensor],
        s: int,
        others: list[int],
        u_e: torch.Tensor,
        h_s_prev: torch.Tensor,
        node_state: dict[int, torch.Tensor],
    ) -> None:
        """A3/A4: updates edge_state IN PLACE for every edge incident to s,
        reading node_state (t-1, unchanged by this call) and each edge's own
        previous value (implicitly zero if the pair has never been touched).
        Non-incident edges (any key not touched here) are left untouched, so
        this alone satisfies "non-incident edges carry over unchanged".
        """
        for j in others:
            key = pair_index(s, j)
            e_prev = edge_state.get(key, u_e.new_zeros(self.EDGE_DIM))
            h_j = node_state[j]
            gru_input = torch.cat([u_e, h_s_prev, h_j]).unsqueeze(0)
            e_new = self.edge_gru(gru_input, e_prev.unsqueeze(0)).squeeze(0)
            edge_state[key] = e_new

    def attend(
        self,
        s: int,
        others: list[int],
        node_state: dict[int, torch.Tensor],
        edge_state: dict[int, torch.Tensor],
        h_s_prev: torch.Tensor,
    ) -> torch.Tensor:
        """A5: edge-conditioned GATv2-style attention context for speaker s,
        aggregating messages from every OTHER known speaker j (no self-loop
        -- edges are only defined for i != j pairs). Reads edge_state "at t"
        (must be called after update_incident_edges for this same s/t).
        Returns a zero vector if `others` is empty (isolated speaker, no
        relationships formed yet -- mirrors the base model's zero-context
        for a brand-new speaker).
        """
        if not others:
            return h_s_prev.new_zeros(self.EDGE_DIM)

        h_j_stack = torch.stack([node_state[j] for j in others])  # [n, node_dim]
        e_sj_stack = torch.stack([edge_state[pair_index(s, j)] for j in others])  # [n, EDGE_DIM]
        h_s_expand = h_s_prev.unsqueeze(0).expand(len(others), -1)

        logits = (self.attn_vec * self.leaky_relu(self.W_n(h_j_stack) + self.W_n(h_s_expand) + self.W_e(e_sj_stack))).sum(
            dim=-1
        )
        alpha = torch.softmax(logits, dim=0)
        messages = self.W_m(torch.cat([h_j_stack, e_sj_stack], dim=-1))
        return (alpha.unsqueeze(-1) * messages).sum(dim=0)

    def edge_mean(
        self, s: int, others: list[int], edge_state: dict[int, torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        """A6: plain (unweighted) mean of speaker s's incident edge states,
        for the classification readout. `reference` supplies dtype/device
        for the zero fallback when `others` is empty.
        """
        if not others:
            return reference.new_zeros(self.EDGE_DIM)
        return torch.stack([edge_state[pair_index(s, j)] for j in others]).mean(dim=0)
