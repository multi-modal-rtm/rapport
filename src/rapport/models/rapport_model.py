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
  Step 4 (temporal=True): adds temporal attention pooling
    (rapport.models.temporal_attention.TemporalAttentionPool) over the
    cached A/V token sequences in place of mean pooling (spec C). Text is
    exempt (already contextual via Phase T). One pool per A/V stream, not
    shared weights (docs/SPEC_RAPPORT_COMPONENTS.md Implementation decisions).

`base_fusion` = relational=False, shift=False, temporal=False (the
ablation matrix's internal baseline, per PHASE N4 Step 1).

Phase N4-R (docs/PHASE_N4R.md, spec v1.1) adds `residual=True`: the whole
multimodal/relational stack now predicts a DELTA on top of the frozen
Phase T text classifier's own logits (`z_text`, cached per utterance),
rather than a logits vector from scratch. `classifier` (renamed `W_out` in
spirit, not in code -- same attribute, same shape, just zero-initialized)
and the fusion layer's audio/video column blocks are both zero-init, so a
FRESH model's predictions equal the Phase T model's exactly at
construction time (`tests/test_rapport_model_residual.py`), regardless of
relational/shift/temporal -- Phase N4's full ablation matrix showed every
added component actively hurt vs. a frozen-text-only anchor
(`docs/PHASE_N4_STEP6.md`), traced to random-init capacity overfitting
(`docs/PHASE_N4R.md` Step 1); residual=True makes "no improvement" the
worst case by construction instead of "regression to below the anchor."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rapport.models.relational_memory import RelationalEdgeMemory
from rapport.models.social_gnn import GraphAttentionLayer
from rapport.models.temporal_attention import TemporalAttentionPool


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
        residual: bool = False,
        gat_heads: int = 4,
        dropout: float = 0.5,
        residual_init_scale: float = 0.0,
        text_feature_dim: int = 768,
    ):
        super().__init__()
        self.relational = relational
        self.shift = shift
        self.temporal = temporal
        self.residual = residual
        self.num_classes = num_classes
        # Second-encoder replication (docs/PREREG_DeBERTa-v3-large.md): the
        # text modality's feature width, independent of FEATURE_DIM (which
        # stays the fixed 768 of the frozen A/V backbones -- unaffected by
        # which text encoder produced text_feat). Default 768 reproduces
        # every existing RoBERTa-base config's fusion layer bit-for-bit.
        self.text_feature_dim = text_feature_dim
        # REVIEWER-RESPONSE control (Major Concern #2): residual_init_scale=0.0
        # (default) is IDENTICAL to the original spec v1.1 behavior below --
        # exact zero-init of W_out and the A/V fusion blocks. A positive value
        # instead draws those same parameters from N(0, residual_init_scale^2),
        # to test whether exact-zero-init under-credits the graph stack via
        # early-epoch gradient starvation / a degenerate local minimum, as
        # opposed to the components genuinely having no value on a fine-tuned
        # foundation. Nothing else about the model changes.
        self.residual_init_scale = residual_init_scale

        self.fusion = nn.Sequential(
            nn.Linear(self.text_feature_dim + 2 * self.FEATURE_DIM, self.FUSION_DIM),
            nn.ReLU(),
        )
        if residual:
            # Phase N4-R v1.1: audio/video enter the fusion projection only
            # as learned corrections -- zero-init their column blocks
            # (text_ctx's block keeps the default Linear init). Bias is
            # NOT zeroed (it isn't tied to any one modality); recorded in
            # docs/SPEC_RAPPORT_COMPONENTS.md's Implementation decisions.
            # Column layout is [text_feature_dim | FEATURE_DIM | FEATURE_DIM]
            # (text, audio, video, matching forward()'s cat order) -- only
            # the audio/video block (columns text_feature_dim: onward) is
            # zeroed, regardless of the text encoder's own width.
            with torch.no_grad():
                if residual_init_scale > 0:
                    self.fusion[0].weight[:, self.text_feature_dim :].normal_(mean=0.0, std=residual_init_scale)
                else:
                    self.fusion[0].weight[:, self.text_feature_dim :].zero_()

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

        if residual:
            # z = z_text + W_out(g_t): W_out (this model's `classifier`,
            # whichever readout dim it has) is zero-init -- weight AND
            # bias -- so a fresh model contributes exactly zero on top of
            # the cached Phase T logits, regardless of relational/shift/
            # temporal or of anything upstream (the rest of the stack can
            # be arbitrarily random at init; W_out=0 erases it entirely).
            # residual_init_scale > 0 (control condition, see above)
            # replaces the exact zero with small-scale random noise instead.
            if residual_init_scale > 0:
                nn.init.normal_(self.classifier.weight, mean=0.0, std=residual_init_scale)
                nn.init.normal_(self.classifier.bias, mean=0.0, std=residual_init_scale)
            else:
                nn.init.zeros_(self.classifier.weight)
                nn.init.zeros_(self.classifier.bias)

        if shift:
            # B3: single-logit linear head on the RAW node state h_{s,t} --
            # not the edge-augmented [h_s || mean_j e_sj] readout the
            # emotion classifier sees, in either path.
            self.shift_head = nn.Linear(self.HIDDEN_DIM, 1)

        if temporal:
            # C1/C4: text_ctx is exempt (its pooling is already contextual --
            # docs/PHASE_N4.md Step 0); separate pool per A/V stream (spec
            # doesn't say to share weights across modalities with different
            # backbone statistics, so this is the more conservative default
            # -- see docs/SPEC_RAPPORT_COMPONENTS.md's Implementation decisions).
            self.video_temporal_pool = TemporalAttentionPool(dim=self.FEATURE_DIM)
            self.audio_temporal_pool = TemporalAttentionPool(dim=self.FEATURE_DIM)

    def _temporal_pool_av(
        self,
        dialogue_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        video_tokens_mask: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_tokens_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools the raw A/V token sequences into [B, L, 768] vectors via
        TemporalAttentionPool, one utterance at a time (no recurrence needed
        for pooling itself). Only computed for positions dialogue_mask says
        are real -- a batch-padding position's token mask is all-False (no
        real tokens were ever written there by collate_dialogues), which
        would make the pool's softmax divide 0/0 (NaN) if fed in directly,
        so those positions are filtered out before pooling and the result
        is scattered back into an all-zero [B, L, 768] tensor (never read
        downstream anyway, since dialogue_mask excludes them everywhere else).
        """
        batch_size, max_len = dialogue_mask.shape
        flat_mask = dialogue_mask.reshape(-1)
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]

        video_tokens_flat = video_tokens.reshape(batch_size * max_len, *video_tokens.shape[2:])[valid_idx]
        video_tokens_mask_flat = video_tokens_mask.reshape(batch_size * max_len, video_tokens_mask.shape[2])[valid_idx]
        pooled_video_valid = self.video_temporal_pool(video_tokens_flat, video_tokens_mask_flat)

        audio_tokens_flat = audio_tokens.reshape(batch_size * max_len, *audio_tokens.shape[2:])[valid_idx]
        audio_tokens_mask_flat = audio_tokens_mask.reshape(batch_size * max_len, audio_tokens_mask.shape[2])[valid_idx]
        pooled_audio_valid = self.audio_temporal_pool(audio_tokens_flat, audio_tokens_mask_flat)

        pooled_video = video_tokens.new_zeros(batch_size * max_len, self.FEATURE_DIM).index_copy(
            0, valid_idx, pooled_video_valid
        )
        pooled_audio = audio_tokens.new_zeros(batch_size * max_len, self.FEATURE_DIM).index_copy(
            0, valid_idx, pooled_audio_valid
        )
        return pooled_video.reshape(batch_size, max_len, self.FEATURE_DIM), pooled_audio.reshape(
            batch_size, max_len, self.FEATURE_DIM
        )

    def forward(
        self,
        video_feat: torch.Tensor,  # [B, L, 768] -- ignored if temporal=True (video_tokens used instead)
        audio_feat: torch.Tensor,  # [B, L, 768] -- ignored if temporal=True (audio_tokens used instead)
        text_feat: torch.Tensor,  # [B, L, 768] -- text_ctx (Phase T contextual cache); always used as-is
        speaker_ids: torch.Tensor,  # [B, L]
        dialogue_mask: torch.Tensor,  # [B, L] bool
        video_tokens: torch.Tensor | None = None,  # [B, L, 392, 768] -- required if temporal=True
        video_tokens_mask: torch.Tensor | None = None,  # [B, L, 392] bool
        audio_tokens: torch.Tensor | None = None,  # [B, L, T_max, 768] -- required if temporal=True
        audio_tokens_mask: torch.Tensor | None = None,  # [B, L, T_max] bool
        text_logits: torch.Tensor | None = None,  # [B, L, num_classes] -- required if residual=True (z_text)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (emotion_logits [B,L,num_classes], shift_logits [B,L] or
        None if shift=False). If residual=True, emotion_logits = z_text +
        W_out(g_t) (spec v1.1) -- added HERE, outside _forward_base/
        _forward_relational, so those two methods (and spec A7's
        bit-for-bit guarantee for _forward_base specifically) never need to
        know about the residual connection at all.
        """
        if self.temporal:
            assert video_tokens is not None and audio_tokens is not None, (
                "temporal=True requires video_tokens/audio_tokens (load_av_tokens=True on the dataset)"
            )
            video_feat, audio_feat = self._temporal_pool_av(
                dialogue_mask, video_tokens, video_tokens_mask, audio_tokens, audio_tokens_mask
            )

        if self.relational:
            logits, shift_logits = self._forward_relational(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)
        else:
            logits, shift_logits = self._forward_base(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)

        if self.residual:
            assert text_logits is not None, "residual=True requires text_logits (the cached Phase T z_text)"
            logits = logits + text_logits

        return logits, shift_logits

    def _forward_base(self, video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask):
        """Step 1 path, UNCHANGED since it was introduced (aside from the
        additive, opt-in shift head below) -- spec A7 requires
        relational=False to reproduce this bit-for-bit, so the emotion-logit
        computation here may not be edited by a later step without
        re-verifying that guarantee.
        """
        batch_size, max_len, _ = video_feat.shape
        fused = self.fusion(torch.cat([text_feat, audio_feat, video_feat], dim=-1))  # [B, L, FUSION_DIM]

        logits = fused.new_zeros(batch_size, max_len, self.classifier.out_features)
        shift_logits = fused.new_zeros(batch_size, max_len) if self.shift else None
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
            if self.shift:
                step_shift_logits = self.shift_head(new_hidden_batch).squeeze(-1)
            for i, b in enumerate(active):
                spk = int(speaker_ids[b, t].item())
                speaker_hidden[b][spk] = new_hidden_batch[i]
                logits[b, t] = step_logits[i]
                if self.shift:
                    shift_logits[b, t] = step_shift_logits[i]

        return logits, shift_logits

    def _forward_relational(self, video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask):
        """Step 2 path (docs/SPEC_RAPPORT_COMPONENTS.md section A). Edge
        states are stored per dialogue in a dict keyed by pair_index over
        LOCAL (per-dialogue, first-appearance-order) speaker indices --
        `local_ids[b]` tracks the global_speaker_id -> local_index mapping,
        assigned lazily the same way `speaker_hidden[b]` grows lazily.
        """
        batch_size, max_len, _ = video_feat.shape
        fused = self.fusion(torch.cat([text_feat, audio_feat, video_feat], dim=-1))  # [B, L, FUSION_DIM]

        logits = fused.new_zeros(batch_size, max_len, self.classifier.out_features)
        shift_logits = fused.new_zeros(batch_size, max_len) if self.shift else None
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
            if self.shift:
                # B3: shift head sees the RAW node state (new_hidden_batch),
                # not the edge-augmented classifier_input_batch.
                step_shift_logits = self.shift_head(new_hidden_batch).squeeze(-1)
            for i, b in enumerate(active):
                logits[b, t] = step_logits[i]
                if self.shift:
                    shift_logits[b, t] = step_shift_logits[i]

        return logits, shift_logits
