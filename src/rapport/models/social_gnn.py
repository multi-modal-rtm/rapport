"""Social GNN — v1 SocialArcNet baseline (ICFNDS '25) conversational classifier.

Given per-utterance frozen V/A/T features, this model:
  1. projects the concatenated 3x768 features to a 256-d fused representation
  2. maintains one hidden state per speaker seen so far in the dialogue
  3. at each utterance step, runs a GAT layer over the *previous* step's
     speaker hidden states (the "social graph"), then updates the current
     speaker's state with a GRUCell fed the fused features + GAT context
  4. classifies the emotion of the current utterance from the updated state

Utterances must be processed in dialogue order (never shuffled) since each
step's graph state depends on all prior steps — MELDCachedDataset/collate
already guarantee this (see rapport.data.meld module docstring).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """Multi-head GAT over a small, fully-connected set of speaker nodes."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes: [N, dim] (N = speakers active so far in this dialogue) -> [N, dim]."""
        n = nodes.shape[0]
        if n == 1:
            return self.out_proj(nodes)

        h = self.proj(nodes).view(n, self.heads, self.head_dim)
        src_score = (h * self.attn_src).sum(-1)  # [N, heads]
        dst_score = (h * self.attn_dst).sum(-1)  # [N, heads]
        scores = self.leaky_relu(src_score.unsqueeze(1) + dst_score.unsqueeze(0))  # [N(dst), N(src), heads]
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)

        h_by_head = h.permute(1, 0, 2)  # [heads, N, head_dim]
        attn_by_head = attn.permute(2, 0, 1)  # [heads, N(dst), N(src)]
        out = torch.bmm(attn_by_head, h_by_head)  # [heads, N, head_dim]
        out = out.permute(1, 0, 2).reshape(n, -1)
        return self.out_proj(out)


class SocialGNN(nn.Module):
    FEATURE_DIM = 768
    FUSION_DIM = 256
    HIDDEN_DIM = 256

    def __init__(self, num_classes: int, gat_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(3 * self.FEATURE_DIM, self.FUSION_DIM),
            nn.ReLU(),
        )
        self.gat = GraphAttentionLayer(self.HIDDEN_DIM, heads=gat_heads, dropout=dropout)
        self.gru_cell = nn.GRUCell(self.FUSION_DIM, self.HIDDEN_DIM)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.HIDDEN_DIM, num_classes)

    def forward(
        self,
        video_feat: torch.Tensor,  # [B, L, 768]
        audio_feat: torch.Tensor,  # [B, L, 768]
        text_feat: torch.Tensor,  # [B, L, 768]
        speaker_ids: torch.Tensor,  # [B, L]
        dialogue_mask: torch.Tensor,  # [B, L] bool
    ) -> torch.Tensor:
        batch_size, max_len, _ = video_feat.shape
        fused = self.fusion(torch.cat([video_feat, audio_feat, text_feat], dim=-1))  # [B, L, FUSION_DIM]

        logits = fused.new_zeros(batch_size, max_len, self.classifier.out_features)
        # One {speaker_id: hidden_state} dict per dialogue in the batch.
        speaker_hidden: list[dict[int, torch.Tensor]] = [{} for _ in range(batch_size)]

        for t in range(max_len):
            active = dialogue_mask[:, t].nonzero(as_tuple=True)[0].tolist()
            if not active:
                continue

            prev_hidden_list = []
            for b in active:
                state = speaker_hidden[b]
                spk = int(speaker_ids[b, t].item())
                if state:
                    node_ids = list(state.keys())
                    nodes = torch.stack([state[s] for s in node_ids])  # [N_b, HIDDEN_DIM]
                    context = self.gat(nodes)
                    prev_hidden = (
                        context[node_ids.index(spk)] if spk in node_ids else fused.new_zeros(self.HIDDEN_DIM)
                    )
                else:
                    prev_hidden = fused.new_zeros(self.HIDDEN_DIM)
                prev_hidden_list.append(prev_hidden)

            prev_hidden_batch = torch.stack(prev_hidden_list)  # [n_active, HIDDEN_DIM]
            fused_batch = fused[active, t]  # [n_active, FUSION_DIM]
            new_hidden_batch = self.gru_cell(fused_batch, prev_hidden_batch)  # [n_active, HIDDEN_DIM]

            step_logits = self.classifier(self.dropout(new_hidden_batch))
            for i, b in enumerate(active):
                spk = int(speaker_ids[b, t].item())
                speaker_hidden[b][spk] = new_hidden_batch[i]
                logits[b, t] = step_logits[i]

        return logits
