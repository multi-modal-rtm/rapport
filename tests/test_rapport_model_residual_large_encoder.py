"""Second-encoder replication (docs/PREREG_DeBERTa-v3-large.md): equality-at-
init for RapportModel's generalized `text_feature_dim` (1024, DeBERTa-v3-large)
-- the same property tests/test_rapport_model_residual.py enforces for the
768-d RoBERTa case, plus a check that the 768-d default path is byte-for-bit
unaffected by the new parameter's existence.
"""

import itertools

import pytest
import torch

from rapport.models.rapport_model import RapportModel

NUM_CLASSES = 7
LARGE_TEXT_DIM = 1024


def _dialogue(length: int, num_speakers: int, seed: int, text_dim: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "video_feat": torch.randn(1, length, 768, generator=g),
        "audio_feat": torch.randn(1, length, 768, generator=g),
        "text_feat": torch.randn(1, length, text_dim, generator=g),
        "speaker_ids": torch.randint(0, num_speakers, (1, length), generator=g),
        "dialogue_mask": torch.ones(1, length, dtype=torch.bool),
        "text_logits": torch.randn(1, length, NUM_CLASSES, generator=g),
    }


@pytest.mark.parametrize("relational,shift,temporal", list(itertools.product([False, True], repeat=3)))
def test_residual_predictions_equal_text_logits_at_init_1024d(relational, shift, temporal):
    d = _dialogue(length=7, num_speakers=3, seed=hash((relational, shift, temporal, "1024")) % (2**31), text_dim=LARGE_TEXT_DIM)
    torch.manual_seed(0)
    model = RapportModel(
        num_classes=NUM_CLASSES, relational=relational, shift=shift, temporal=temporal, residual=True,
        text_feature_dim=LARGE_TEXT_DIM,
    )
    model.eval()

    kwargs = {"text_logits": d["text_logits"]}
    if temporal:
        video_tokens = torch.randn(1, 7, 392, 768)
        audio_tokens = torch.randn(1, 7, 20, 768)
        kwargs.update(
            video_tokens=video_tokens,
            video_tokens_mask=torch.ones(1, 7, 392, dtype=torch.bool),
            audio_tokens=audio_tokens,
            audio_tokens_mask=torch.ones(1, 7, 20, dtype=torch.bool),
        )

    with torch.no_grad():
        logits, _ = model(
            d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"], **kwargs
        )

    assert torch.equal(logits, d["text_logits"]), (
        f"relational={relational} shift={shift} temporal={temporal} text_feature_dim={LARGE_TEXT_DIM}: "
        "fresh residual model's logits must exactly equal the cached large-encoder text logits at init"
    )


def test_fusion_shape_and_zero_init_columns_for_1024d_text():
    model = RapportModel(num_classes=NUM_CLASSES, residual=True, text_feature_dim=LARGE_TEXT_DIM)
    w = model.fusion[0].weight
    assert w.shape[1] == LARGE_TEXT_DIM + 2 * 768
    assert torch.equal(w[:, LARGE_TEXT_DIM:], torch.zeros_like(w[:, LARGE_TEXT_DIM:])), (
        "audio/video column blocks must be zero-init regardless of text_feature_dim"
    )
    assert not torch.equal(w[:, :LARGE_TEXT_DIM], torch.zeros_like(w[:, :LARGE_TEXT_DIM])), (
        "text column block must NOT be zero-init"
    )


def test_default_text_feature_dim_unchanged_from_768():
    """The new parameter must not perturb the existing RoBERTa-base (768-d)
    fusion layer's shape -- default construction is identical to before this
    parameter existed."""
    model = RapportModel(num_classes=NUM_CLASSES, residual=True)
    assert model.text_feature_dim == 768
    assert model.fusion[0].weight.shape[1] == 3 * 768
