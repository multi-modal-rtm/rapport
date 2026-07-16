"""Phase N4-R Step 3 property test (spec v1.1): at initialization, EVERY
config's predictions must equal the Phase T model's own predictions
(z_text) exactly -- W_out (the classifier) is zero-init, so the rest of
the stack (arbitrarily random at init, including random A/V input) must
contribute exactly zero on top of z_text, regardless of relational/shift/
temporal.
"""

import itertools

import pytest
import torch

from rapport.models.rapport_model import RapportModel

NUM_CLASSES = 7
VIDEO_TOKEN_LEN = 392


def _dialogue(length: int, num_speakers: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "video_feat": torch.randn(1, length, 768, generator=g),
        "audio_feat": torch.randn(1, length, 768, generator=g),
        "text_feat": torch.randn(1, length, 768, generator=g),
        "speaker_ids": torch.randint(0, num_speakers, (1, length), generator=g),
        "dialogue_mask": torch.ones(1, length, dtype=torch.bool),
        "video_tokens": torch.randn(1, length, VIDEO_TOKEN_LEN, 768, generator=g),
        "video_tokens_mask": torch.ones(1, length, VIDEO_TOKEN_LEN, dtype=torch.bool),
        "audio_tokens": torch.randn(1, length, 20, 768, generator=g),
        "audio_tokens_mask": torch.ones(1, length, 20, dtype=torch.bool),
        "text_logits": torch.randn(1, length, NUM_CLASSES, generator=g),
    }


@pytest.mark.parametrize("relational,shift,temporal", list(itertools.product([False, True], repeat=3)))
def test_residual_predictions_equal_text_logits_at_init(relational, shift, temporal):
    d = _dialogue(length=7, num_speakers=3, seed=hash((relational, shift, temporal)) % (2**31))
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=relational, shift=shift, temporal=temporal, residual=True)
    model.eval()

    kwargs = {"text_logits": d["text_logits"]}
    if temporal:
        kwargs.update(
            video_tokens=d["video_tokens"], video_tokens_mask=d["video_tokens_mask"],
            audio_tokens=d["audio_tokens"], audio_tokens_mask=d["audio_tokens_mask"],
        )

    with torch.no_grad():
        logits, _ = model(
            d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"], **kwargs
        )

    assert torch.equal(logits, d["text_logits"]), (
        f"relational={relational} shift={shift} temporal={temporal}: fresh residual model's logits must exactly "
        "equal the cached Phase T text logits at initialization"
    )


def test_residual_predictions_differ_from_text_logits_after_training():
    """Sanity check for the property test's own meaningfulness: after even a
    few optimizer steps, the residual model's logits DO diverge from
    z_text (the correction is learnable, just zero at t=0).
    """
    d = _dialogue(length=6, num_speakers=2, seed=1)
    labels = torch.randint(0, NUM_CLASSES, (1, 6))
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, residual=True, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(10):
        optimizer.zero_grad()
        logits, _ = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"], text_logits=d["text_logits"])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"], text_logits=d["text_logits"])

    assert not torch.allclose(logits, d["text_logits"]), "after training, the residual correction should be nonzero"


def test_residual_gradient_flows_to_all_parameters():
    d = _dialogue(length=6, num_speakers=2, seed=2)
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True, shift=True, residual=True)

    logits, shift_logits = model(
        d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"], text_logits=d["text_logits"]
    )
    (logits.sum() + shift_logits.sum()).backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient at {name}"


def test_fusion_av_blocks_are_zero_init_when_residual():
    model = RapportModel(num_classes=NUM_CLASSES, residual=True)
    w = model.fusion[0].weight
    assert torch.equal(w[:, 768:], torch.zeros_like(w[:, 768:])), "audio/video column blocks must be zero-init"
    assert not torch.equal(w[:, :768], torch.zeros_like(w[:, :768])), "text_ctx column block must NOT be zero-init"


def test_fusion_av_blocks_are_not_zero_init_when_not_residual():
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, residual=False)
    w = model.fusion[0].weight
    assert not torch.equal(w[:, 768:], torch.zeros_like(w[:, 768:])), "non-residual path must keep normal A/V init"
