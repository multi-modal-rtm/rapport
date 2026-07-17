"""Phase N4-R Step 3 property test (spec v1.1): at initialization, EVERY
config's predictions must equal the Phase T model's own predictions
(z_text) exactly -- W_out (the classifier) is zero-init, so the rest of
the stack (arbitrarily random at init, including random A/V input) must
contribute exactly zero on top of z_text, regardless of relational/shift/
temporal.
"""

import itertools
from pathlib import Path

import pandas as pd
import pytest
import torch

from rapport.models.rapport_model import RapportModel

NUM_CLASSES = 7
VIDEO_TOKEN_LEN = 392

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEXT_LOGITS_DIR = PROJECT_ROOT / "data" / "meld" / "cache" / "text_logits"
FROZEN_TEXT_FEAT_DIR = PROJECT_ROOT / "data" / "meld" / "cache" / "text"
PROCESSED_DIR = PROJECT_ROOT / "data" / "meld" / "processed"


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


@pytest.mark.slow
@pytest.mark.skipif(
    not FROZEN_TEXT_LOGITS_DIR.exists(),
    reason="requires the BRIDGING EXPERIMENT's text_logits cache (scripts/train_frozen_text_foundation.py)",
)
@pytest.mark.parametrize("relational,shift,temporal", [(False, False, False), (True, True, True)])
def test_residual_equals_frozen_text_foundation_logits_on_real_cache(relational, shift, temporal):
    """Same property as test_residual_predictions_equal_text_logits_at_init,
    but against REAL data: one dialogue's actual frozen (non-contextual)
    text features and the frozen-era foundation's actual cached logits
    (BRIDGING EXPERIMENT step 1/2, docs/PHASE_N5A.md), for the two configs
    the bridging matrix actually trains (base_fusion_R and full_R)."""
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    dialogue_id = int(df["dialogue_id"].iloc[0])
    rows = df[df["dialogue_id"] == dialogue_id].sort_values("utterance_id")
    assert len(rows) > 0

    video_feat, audio_feat, text_feat, text_logits = [], [], [], []
    for row in rows.itertuples(index=False):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        video_feat.append(torch.load(FROZEN_TEXT_FEAT_DIR.parent / "video" / "test" / stem, weights_only=True))
        audio_feat.append(torch.load(FROZEN_TEXT_FEAT_DIR.parent / "audio" / "test" / stem, weights_only=True))
        text_feat.append(torch.load(FROZEN_TEXT_FEAT_DIR / "test" / stem, weights_only=True))
        text_logits.append(torch.load(FROZEN_TEXT_LOGITS_DIR / "test" / stem, weights_only=True))

    length = len(rows)
    d = {
        "video_feat": torch.stack(video_feat).unsqueeze(0),
        "audio_feat": torch.stack(audio_feat).unsqueeze(0),
        "text_feat": torch.stack(text_feat).unsqueeze(0),
        "speaker_ids": torch.tensor(rows["speaker_id"].tolist(), dtype=torch.long).unsqueeze(0),
        "dialogue_mask": torch.ones(1, length, dtype=torch.bool),
        "text_logits": torch.stack(text_logits).unsqueeze(0),
    }

    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=relational, shift=shift, temporal=temporal, residual=True)
    model.eval()

    kwargs = {"text_logits": d["text_logits"]}
    if temporal:
        video_tokens, audio_tokens = [], []
        for row in rows.itertuples(index=False):
            stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
            video_tokens.append(torch.load(FROZEN_TEXT_FEAT_DIR.parent / "video_tokens" / "test" / stem, weights_only=True))
            audio_tokens.append(torch.load(FROZEN_TEXT_FEAT_DIR.parent / "audio_tokens" / "test" / stem, weights_only=True))
        audio_max_len = max(a.shape[0] for a in audio_tokens)
        audio_padded = torch.stack(
            [torch.nn.functional.pad(a, (0, 0, 0, audio_max_len - a.shape[0])) for a in audio_tokens]
        ).unsqueeze(0)
        audio_mask = torch.zeros(1, length, audio_max_len, dtype=torch.bool)
        for i, a in enumerate(audio_tokens):
            audio_mask[0, i, : a.shape[0]] = True
        kwargs.update(
            video_tokens=torch.stack(video_tokens).unsqueeze(0),
            video_tokens_mask=torch.ones(1, length, VIDEO_TOKEN_LEN, dtype=torch.bool),
            audio_tokens=audio_padded,
            audio_tokens_mask=audio_mask,
        )

    with torch.no_grad():
        logits, _ = model(
            d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"], **kwargs
        )

    assert torch.equal(logits, d["text_logits"]), (
        f"relational={relational} shift={shift} temporal={temporal}: fresh residual model's logits must exactly "
        "equal the frozen-era foundation's cached logits at initialization, on real cached data"
    )
