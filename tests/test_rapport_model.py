import torch
import torch.nn.functional as F

from rapport.models.rapport_model import RapportModel

NUM_CLASSES = 7


def _synthetic_batch(seed: int = 0):
    """One dialogue, 2 speakers alternating, 6 utterances. Video/audio are
    ZEROED -- all class signal must come through text_feat alone, exercising
    the "text_ctx-only" pathway the Phase N4 Step 1 unit test is about.
    """
    g = torch.Generator().manual_seed(seed)
    length = 6
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    # distinct, class-correlated text features so the model has real signal to fit
    class_prototypes = torch.randn(NUM_CLASSES, 768, generator=g)
    text_feat = class_prototypes[labels] + 0.01 * torch.randn(length, 768, generator=g)

    video_feat = torch.zeros(1, length, 768)
    audio_feat = torch.zeros(1, length, 768)
    text_feat = text_feat.unsqueeze(0)  # [1, L, 768]
    speaker_ids = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.long)
    dialogue_mask = torch.ones(1, length, dtype=torch.bool)
    labels = labels.unsqueeze(0)  # [1, L]

    return video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask, labels


def test_base_fusion_forward_is_finite_with_av_zeroed():
    video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask, labels = _synthetic_batch()
    model = RapportModel(num_classes=NUM_CLASSES)
    logits = model(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)

    assert logits.shape == (1, 6, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_base_fusion_trains_on_text_ctx_alone():
    """The fusion/GAT/GRU pathway must be sane and trainable even when audio
    and video contribute nothing (all-zero) -- text_ctx is the only signal.
    """
    video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask, labels = _synthetic_batch()
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        logits = model(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)
        loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_preds = logits.argmax(dim=-1)
    train_acc = (final_preds == labels).float().mean().item()

    assert losses[-1] < losses[0], "loss should decrease over training"
    assert train_acc == 1.0, f"model should perfectly fit this tiny synthetic batch, got acc={train_acc}"


def test_base_fusion_bit_for_bit_reproducible_given_fixed_seed_and_weights():
    video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask, _ = _synthetic_batch()
    torch.manual_seed(0)
    model_a = RapportModel(num_classes=NUM_CLASSES)
    torch.manual_seed(0)
    model_b = RapportModel(num_classes=NUM_CLASSES)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)
        out_b = model_b(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)

    assert torch.equal(out_a, out_b)


def test_not_implemented_flags_raise_clearly():
    import pytest

    with pytest.raises(NotImplementedError):
        RapportModel(num_classes=NUM_CLASSES, shift=True)
    with pytest.raises(NotImplementedError):
        RapportModel(num_classes=NUM_CLASSES, temporal=True)
