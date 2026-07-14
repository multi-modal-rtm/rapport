"""RAPPORT core model: multimodal fusion + (optionally) relational edge
memory + emotion-shift auxiliary head + temporal attention pooling.

Built incrementally across Phase N4's steps (docs/PHASE_N4.md):
  Step 1 (this file, base_fusion): U_e = ReLU(W_proj[text_ctx||A||V]+b),
    then the v1-style per-speaker GAT+GRUCell node update (identical
    mechanics to rapport.models.social_gnn.SocialGNN, just fed the
    Phase T contextual text cache instead of the frozen non-contextual
    one) and a linear classifier.
  Step 2 (relational=True): adds edge-state relational memory
    (docs/SPEC_RAPPORT_COMPONENTS.md section A). relational=False must
    reproduce this step's base_fusion path bit-for-bit (spec A7) -- so the
    non-relational path here is never touched again once Step 2 lands.
  Step 3 (shift=True): adds an emotion-shift auxiliary head (spec B).
  Step 4 (temporal=True): adds temporal attention pooling over the cached
    A/V token sequences in place of mean pooling (spec C).

`base_fusion` = relational=False, shift=False, temporal=False (the
ablation matrix's internal baseline, per PHASE N4 Step 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rapport.models.social_gnn import GraphAttentionLayer


class RapportModel(nn.Module):
    FEATURE_DIM = 768
    FUSION_DIM = 256
    HIDDEN_DIM = 256

    def __init__(
        self,
        num_classes: int,
        relational: bool = False,
        shift: bool = False,
        temporal: bool = False,
        gat_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        if relational:
            raise NotImplementedError("relational=True lands in Phase N4 Step 2 (docs/PHASE_N4.md)")
        if shift:
            raise NotImplementedError("shift=True lands in Phase N4 Step 3 (docs/PHASE_N4.md)")
        if temporal:
            raise NotImplementedError("temporal=True lands in Phase N4 Step 4 (docs/PHASE_N4.md)")

        self.relational = relational
        self.shift = shift
        self.temporal = temporal

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
        text_feat: torch.Tensor,  # [B, L, 768] -- text_ctx (Phase T contextual cache)
        speaker_ids: torch.Tensor,  # [B, L]
        dialogue_mask: torch.Tensor,  # [B, L] bool
    ) -> torch.Tensor:
        batch_size, max_len, _ = video_feat.shape
        fused = self.fusion(torch.cat([text_feat, audio_feat, video_feat], dim=-1))  # [B, L, FUSION_DIM]

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
