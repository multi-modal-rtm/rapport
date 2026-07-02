"""Frozen v1 backbones: MViTv2 (video), Wav2Vec 2.0 base (audio), RoBERTa base (text).

Each wrapper exposes a uniform `forward(...) -> (pooled, tokens)` contract:
  - pooled: [B, 768] — mean over the temporal/sequence dimension for video and
    audio, the <s> (CLS) token of `last_hidden_state` for text.
  - tokens: [B, N, 768] — the full pre-pooling token sequence, kept around for
    the temporal-attention configs (B, D) added in v2.

Text note: `RobertaModel`'s `pooler_output` (Dense+Tanh over the [CLS]
token) is randomly re-initialized on every process, since `roberta-base`'s
checkpoint ships without pretrained pooler weights (unlike BERT). We
deliberately do NOT use it — see `RobertaBackbone` below.

All three are frozen (`requires_grad=False`) in this phase; nothing here calls
`torch.no_grad()` internally so the same wrappers can be reused, un-frozen,
for the LoRA-adapted regime later. Callers building the frozen-feature cache
should wrap their calls in `torch.no_grad()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
from transformers import RobertaModel, Wav2Vec2Model


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


class MViTv2Backbone(nn.Module):
    """Wraps torchvision's mvit_v2_s, exposing pre-head tokens alongside the pooled output.

    torchvision's public `MViT.forward` only returns classification logits, so
    we replicate its internals (conv_proj -> pos_encoding -> blocks -> norm)
    to recover the token sequence before the classifier head is applied.
    """

    OUTPUT_DIM = 768
    # The Kinetics-pretrained positional encoding is fixed for 16-frame clips
    # (torchvision's MViT_V2_S_Weights meta["min_temporal_size"] == 16); our
    # preprocessing stores 8 uniformly-sampled frames per utterance, so we
    # temporally repeat-interleave (8 -> 16) before the model's patch embed.
    PRETRAINED_NUM_FRAMES = 16

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.model = mvit_v2_s(weights=weights)
        _freeze(self.model)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames: [B, 8, 3, 224, 224] -> pooled [B, 768], tokens [B, N_patches, 768]."""
        num_frames = frames.shape[1]
        if num_frames != self.PRETRAINED_NUM_FRAMES:
            assert self.PRETRAINED_NUM_FRAMES % num_frames == 0, (
                f"{num_frames} frames must evenly divide into {self.PRETRAINED_NUM_FRAMES}"
            )
            frames = frames.repeat_interleave(self.PRETRAINED_NUM_FRAMES // num_frames, dim=1)

        m = self.model
        x = frames.permute(0, 2, 1, 3, 4)  # [B, 3, 16, 224, 224] (channels-first for Conv3d)
        x = m.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = m.pos_encoding(x)
        thw = (m.pos_encoding.temporal_size,) + m.pos_encoding.spatial_size
        for block in m.blocks:
            x, thw = block(x, thw)
        x = m.norm(x)

        tokens = x[:, 1:]  # drop the class token
        pooled = tokens.mean(dim=1)
        return pooled, tokens


class Wav2Vec2Backbone(nn.Module):
    """Wraps HF Wav2Vec2Model (base, SSL-pretrained, no ASR fine-tuning)."""

    OUTPUT_DIM = 768

    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        _freeze(self.model)

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input_values: [B, T_samples] raw 16kHz waveform -> pooled [B, 768], tokens [B, T', 768]."""
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state

        if attention_mask is not None:
            feat_mask = self.model._get_feature_vector_attention_mask(tokens.shape[1], attention_mask)
            mask = feat_mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = tokens.mean(dim=1)
        return pooled, tokens


class RobertaBackbone(nn.Module):
    """Wraps HF RobertaModel, using the raw pretrained <s> (CLS) token as the
    pooled output rather than `pooler_output`.

    `RobertaModel(add_pooling_layer=True)`'s pooler (Dense -> Tanh over the
    [CLS] hidden state) is randomly re-initialized every process — verified
    directly in docs/DIAGNOSIS.md: two fresh instances given identical input
    produce outputs differing by up to 0.93, and a cache built across a
    crash/resume process boundary showed near-orthogonal (cosine ~0.015)
    features between segments. `last_hidden_state[:, 0]` is the actual
    pretrained, deterministic, process-independent <s> token representation
    and carries no such hazard.
    """

    OUTPUT_DIM = 768

    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        _freeze(self.model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input_ids: [B, L] -> pooled [B, 768] (<s> token), tokens [B, L, 768]."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state
        pooled = tokens[:, 0]
        return pooled, tokens
