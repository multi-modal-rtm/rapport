"""RAPPORT core model: multimodal fusion + (optionally) relational edge
memory + emotion-shift auxiliary head + temporal attention pooling.

Built incrementally across Phase N4's steps (docs/PHASE_N4.md):
  Step 1 (base_fusion): U_e = ReLU(W_proj[text_ctx||A||V]+b), then the
    v1-style per-speaker GAT+GRUCell node update (identical mechanics to
    rapport.models.social_gnn.SocialGNN, just fed the Phase T contextual
    text cache instead of the frozen non-contextual one) and a linear
    classifier.
  Step 2 (this file, relational=True): adds edge-state relational memory
    (docs/SPEC_RAPPORT_COMPONENTS.md section A, rapport.models.relational_memory).
    relational=False reproduces Step 1's base_fusion path bit-for-bit (spec
    A7) -- the non-relational branch below is never touched by this step.
  Step 3 (shift=True): adds an emotion-shift auxiliary head (spec B).
  Step 4 (temporal=True): adds temporal attention pooling over the cached
    A/V token sequences in place of mean pooling (spec C).

`base_fusion` = relational=False, shift=False, temporal=False (the
ablation matrix's internal baseline, per PHASE N4 Step 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rapport.models.relational_memory import RelationalEdgeMemory
from rapport.models.social_gnn import GraphAttentionLayer


class RapportModel(nn.Module):
    FEATURE_DIM = 768
    FUSION_DIM = 256
    HIDDEN_DIM = 256
    EDGE_DIM = RelationalEdgeMemory.EDGE_DIM

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
        self.dropout = nn.Dropout(dropout)

        if relational:
            # A5/A6: node update reads edge-conditioned attention context
            # (fed as extra GRU INPUT alongside the utterance features,
            # while the speaker's own raw previous state remains the GRU
            # HIDDEN argument -- edges never carry self-information, so
            # "my own history" and "what I'm picking up about my
            # relationships" are kept as separate signals rather than
            # conflated through a single self-attending GAT the way the
            # non-relational path does it. Recorded in
            # docs/SPEC_RAPPORT_COMPONENTS.md's Implementation decisions,
            # since the spec doesn't pin down this exact wiring.)
            self.rel_mem = RelationalEdgeMemory(node_dim=self.HIDDEN_DIM, utt_dim=self.FUSION_DIM)
            self.rel_gru_cell = nn.GRUCell(self.FUSION_DIM + self.EDGE_DIM, self.HIDDEN_DIM)
            self.classifier = nn.Linear(self.HIDDEN_DIM + self.EDGE_DIM, num_classes)
        else:
            self.gat = GraphAttentionLayer(self.HIDDEN_DIM, heads=gat_heads, dropout=dropout)
            self.gru_cell = nn.GRUCell(self.FUSION_DIM, self.HIDDEN_DIM)
            self.classifier = nn.Linear(self.HIDDEN_DIM, num_classes)

    def forward(
        self,
        video_feat: torch.Tensor,  # [B, L, 768]
        audio_feat: torch.Tensor,  # [B, L, 768]
        text_feat: torch.Tensor,  # [B, L, 768] -- text_ctx (Phase T contextual cache)
        speaker_ids: torch.Tensor,  # [B, L]
        dialogue_mask: torch.Tensor,  # [B, L] bool
    ) -> torch.Tensor:
        if self.relational:
            return self._forward_relational(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)
        return self._forward_base(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)

    def _forward_base(self, video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask) -> torch.Tensor:
        """Step 1 path, UNCHANGED since it was introduced -- spec A7 requires
        relational=False to reproduce this bit-for-bit, so nothing here may
        be edited by a later step without re-verifying that guarantee.
        """
        batch_size, max_len, _ = video_feat.shape
        fused = self.fusion(torch.cat([text_feat, audio_feat, video_feat], dim=-1))  # [B, L, FUSION_DIM]

        logits = fused.new_zeros(batch_size, max_len, self.classifier.out_features)
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

    def _forward_relational(self, video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask) -> torch.Tensor:
        """Step 2 path (docs/SPEC_RAPPORT_COMPONENTS.md section A). Edge
        states are stored per dialogue in a dict keyed by pair_index over
        LOCAL (per-dialogue, first-appearance-order) speaker indices --
        `local_ids[b]` tracks the global_speaker_id -> local_index mapping,
        assigned lazily the same way `speaker_hidden[b]` grows lazily.
        """
        batch_size, max_len, _ = video_feat.shape
        fused = self.fusion(torch.cat([text_feat, audio_feat, video_feat], dim=-1))  # [B, L, FUSION_DIM]

        logits = fused.new_zeros(batch_size, max_len, self.classifier.out_features)
        speaker_hidden: list[dict[int, torch.Tensor]] = [{} for _ in range(batch_size)]
        edge_hidden: list[dict[int, torch.Tensor]] = [{} for _ in range(batch_size)]
        local_ids: list[dict[int, int]] = [{} for _ in range(batch_size)]

        for t in range(max_len):
            active = dialogue_mask[:, t].nonzero(as_tuple=True)[0].tolist()
            if not active:
                continue

            gru_input_list, h_prev_list, spk_list = [], [], []
            for b in active:
                state = speaker_hidden[b]
                local_id = local_ids[b]
                edge_state = edge_hidden[b]
                spk = int(speaker_ids[b, t].item())
                if spk not in local_id:
                    local_id[spk] = len(local_id)

                others_global = [g for g in state.keys() if g != spk]
                others_local = [local_id[g] for g in others_global]
                local_node_state = {local_id[g]: state[g] for g in others_global}
                h_s_prev = state.get(spk, fused.new_zeros(self.HIDDEN_DIM))
                u_e = fused[b, t]

                # A4: edges read node states at t-1 (local_node_state/h_s_prev,
                # both pre-update) -- must happen before attend() below.
                self.rel_mem.update_incident_edges(
                    edge_state, local_id[spk], others_local, u_e, h_s_prev, local_node_state
                )
                # A4/A5: node update reads edge states at t (just written above).
                context = self.rel_mem.attend(local_id[spk], others_local, local_node_state, edge_state, h_s_prev)

                gru_input_list.append(torch.cat([u_e, context]))
                h_prev_list.append(h_s_prev)
                spk_list.append(spk)

            gru_input_batch = torch.stack(gru_input_list)
            h_prev_batch = torch.stack(h_prev_list)
            new_hidden_batch = self.rel_gru_cell(gru_input_batch, h_prev_batch)  # [n_active, HIDDEN_DIM]

            classifier_input_list = []
            for i, b in enumerate(active):
                state = speaker_hidden[b]
                local_id = local_ids[b]
                edge_state = edge_hidden[b]
                spk = spk_list[i]
                others_local = [local_id[g] for g in state.keys() if g != spk]

                # A6: readout from [h_{s,t} || mean_j e_{sj,t}].
                edge_mean = self.rel_mem.edge_mean(local_id[spk], others_local, edge_state, new_hidden_batch[i])
                classifier_input_list.append(torch.cat([new_hidden_batch[i], edge_mean]))
                state[spk] = new_hidden_batch[i]

            classifier_input_batch = torch.stack(classifier_input_list)
            step_logits = self.classifier(self.dropout(classifier_input_batch))
            for i, b in enumerate(active):
                logits[b, t] = step_logits[i]

        return logits
