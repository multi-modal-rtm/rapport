"""Model-level temporal-attention integration tests (docs/SPEC_RAPPORT_COMPONENTS.md
section C4 -- the pooling module itself is tested standalone in
tests/test_temporal_attention.py).
"""

import torch

from rapport.models.rapport_model import RapportModel

NUM_CLASSES = 7
VIDEO_TOKEN_LEN = 392


def _dialogue_with_tokens(length: int, num_speakers: int, seed: int, audio_token_len: int = 20) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "video_feat": torch.randn(1, length, 768, generator=g),  # unused when temporal=True, but still passed
        "audio_feat": torch.randn(1, length, 768, generator=g),
        "text_feat": torch.randn(1, length, 768, generator=g),
        "speaker_ids": torch.randint(0, num_speakers, (1, length), generator=g),
        "dialogue_mask": torch.ones(1, length, dtype=torch.bool),
        "video_tokens": torch.randn(1, length, VIDEO_TOKEN_LEN, 768, generator=g),
        "video_tokens_mask": torch.ones(1, length, VIDEO_TOKEN_LEN, dtype=torch.bool),
        "audio_tokens": torch.randn(1, length, audio_token_len, 768, generator=g),
        "audio_tokens_mask": torch.ones(1, length, audio_token_len, dtype=torch.bool),
    }


def test_temporal_forward_is_finite():
    d = _dialogue_with_tokens(length=6, num_speakers=3, seed=0)
    model = RapportModel(num_classes=NUM_CLASSES, temporal=True)
    logits, _ = model(
        d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"],
        video_tokens=d["video_tokens"], video_tokens_mask=d["video_tokens_mask"],
        audio_tokens=d["audio_tokens"], audio_tokens_mask=d["audio_tokens_mask"],
    )
    assert logits.shape == (1, 6, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_temporal_works_with_relational_and_shift():
    d = _dialogue_with_tokens(length=6, num_speakers=3, seed=1)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True, shift=True, temporal=True)
    logits, shift_logits = model(
        d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"],
        video_tokens=d["video_tokens"], video_tokens_mask=d["video_tokens_mask"],
        audio_tokens=d["audio_tokens"], audio_tokens_mask=d["audio_tokens_mask"],
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(shift_logits).all()


def test_temporal_handles_batch_padding_without_nan():
    """A batch-padding dialogue position (from a shorter dialogue in the same
    batch as a longer one) has an all-False token mask -- must not produce
    NaN (which would otherwise contaminate every other row's gradient via
    shared temporal-pool parameters).
    """
    short = _dialogue_with_tokens(length=4, num_speakers=2, seed=2)
    long = _dialogue_with_tokens(length=7, num_speakers=2, seed=3)

    max_len = 7
    pad = max_len - 4

    def pad_dialogue_mask(d, length, max_len):
        mask = torch.zeros(1, max_len, dtype=torch.bool)
        mask[0, :length] = True
        return mask

    video_tokens = torch.cat([short["video_tokens"], torch.zeros(1, pad, VIDEO_TOKEN_LEN, 768)], dim=1)
    video_tokens_mask = torch.cat([short["video_tokens_mask"], torch.zeros(1, pad, VIDEO_TOKEN_LEN, dtype=torch.bool)], dim=1)
    audio_tokens = torch.cat([short["audio_tokens"], torch.zeros(1, pad, 20, 768)], dim=1)
    audio_tokens_mask = torch.cat([short["audio_tokens_mask"], torch.zeros(1, pad, 20, dtype=torch.bool)], dim=1)
    dialogue_mask = pad_dialogue_mask(short, 4, max_len)
    speaker_ids = torch.cat([short["speaker_ids"], torch.zeros(1, pad, dtype=torch.long)], dim=1)
    video_feat = torch.cat([short["video_feat"], torch.zeros(1, pad, 768)], dim=1)
    audio_feat = torch.cat([short["audio_feat"], torch.zeros(1, pad, 768)], dim=1)
    text_feat = torch.cat([short["text_feat"], torch.zeros(1, pad, 768)], dim=1)

    batch = {
        "video_feat": torch.cat([video_feat, long["video_feat"]], dim=0),
        "audio_feat": torch.cat([audio_feat, long["audio_feat"]], dim=0),
        "text_feat": torch.cat([text_feat, long["text_feat"]], dim=0),
        "speaker_ids": torch.cat([speaker_ids, long["speaker_ids"]], dim=0),
        "dialogue_mask": torch.cat([dialogue_mask, long["dialogue_mask"]], dim=0),
        "video_tokens": torch.cat([video_tokens, long["video_tokens"]], dim=0),
        "video_tokens_mask": torch.cat([video_tokens_mask, long["video_tokens_mask"]], dim=0),
        "audio_tokens": torch.cat([audio_tokens, long["audio_tokens"]], dim=0),
        "audio_tokens_mask": torch.cat([audio_tokens_mask, long["audio_tokens_mask"]], dim=0),
    }

    model = RapportModel(num_classes=NUM_CLASSES, temporal=True)
    logits, _ = model(
        batch["video_feat"], batch["audio_feat"], batch["text_feat"], batch["speaker_ids"], batch["dialogue_mask"],
        video_tokens=batch["video_tokens"], video_tokens_mask=batch["video_tokens_mask"],
        audio_tokens=batch["audio_tokens"], audio_tokens_mask=batch["audio_tokens_mask"],
    )
    assert torch.isfinite(logits).all(), "padded dialogue positions must not produce NaN"

    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite (NaN?) gradient at {name} -- padding leaked in"


def test_gradient_flows_to_temporal_pool_parameters():
    d = _dialogue_with_tokens(length=5, num_speakers=2, seed=4)
    model = RapportModel(num_classes=NUM_CLASSES, temporal=True)
    logits, _ = model(
        d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"],
        video_tokens=d["video_tokens"], video_tokens_mask=d["video_tokens_mask"],
        audio_tokens=d["audio_tokens"], audio_tokens_mask=d["audio_tokens_mask"],
    )
    logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
