"""Model-level shift-head tests (docs/SPEC_RAPPORT_COMPONENTS.md section B,
remaining B6 items -- label derivation itself is covered in
tests/test_shift_labels.py).
"""

import torch
import torch.nn.functional as F

from rapport.models.rapport_model import RapportModel

NUM_CLASSES = 7


def _dialogue(length: int, num_speakers: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "video_feat": torch.randn(1, length, 768, generator=g),
        "audio_feat": torch.randn(1, length, 768, generator=g),
        "text_feat": torch.randn(1, length, 768, generator=g),
        "speaker_ids": torch.randint(0, num_speakers, (1, length), generator=g),
        "dialogue_mask": torch.ones(1, length, dtype=torch.bool),
    }


def test_shift_logits_shape_when_enabled():
    d = _dialogue(length=8, num_speakers=3, seed=0)
    model = RapportModel(num_classes=NUM_CLASSES, shift=True)
    logits, shift_logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
    assert logits.shape == (1, 8, NUM_CLASSES)
    assert shift_logits.shape == (1, 8)
    assert torch.isfinite(shift_logits).all()


def test_shift_logits_none_when_disabled():
    d = _dialogue(length=8, num_speakers=3, seed=0)
    model = RapportModel(num_classes=NUM_CLASSES, shift=False)
    _, shift_logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
    assert shift_logits is None


def test_shift_head_works_with_relational_true():
    d = _dialogue(length=8, num_speakers=3, seed=1)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True, shift=True)
    logits, shift_logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
    assert torch.isfinite(logits).all()
    assert torch.isfinite(shift_logits).all()


def test_masked_shift_positions_contribute_zero_gradient():
    """B6: a masked position's shift_label value must never influence the
    loss or gradient -- verified by changing ONLY a masked position's label
    and confirming the loss/gradients are bit-identical.
    """
    d = _dialogue(length=6, num_speakers=2, seed=2)
    torch.manual_seed(0)
    # dropout=0.0: this test compares two separate forward passes for exact
    # equality, which dropout's per-call randomness would otherwise break
    # regardless of the masking behavior under test.
    model = RapportModel(num_classes=NUM_CLASSES, shift=True, dropout=0.0)
    model.eval()

    shift_label_a = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    shift_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])  # position 0 is masked (e.g. first utterance)
    # Position 0 flipped (0.0 -> 1.0) -- since it's masked, this must not matter.
    shift_label_b = shift_label_a.clone()
    shift_label_b[0, 0] = 1.0

    def masked_shift_loss(shift_label):
        model.zero_grad()
        _, shift_logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
        valid = d["dialogue_mask"] & (shift_mask > 0.5)
        loss = F.binary_cross_entropy_with_logits(shift_logits[valid], shift_label[valid])
        loss.backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        return loss.item(), grads

    loss_a, grads_a = masked_shift_loss(shift_label_a)
    loss_b, grads_b = masked_shift_loss(shift_label_b)

    assert loss_a == loss_b
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), f"gradient at {name} differs after changing a MASKED position"


def test_combined_ce_and_shift_loss_trains():
    """B4: L = L_CE_emotion + 0.5*L_shift must be a sane, trainable combined
    objective -- fit a tiny synthetic batch with both signals present.
    """
    length = 6
    g = torch.Generator().manual_seed(0)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    class_prototypes = torch.randn(NUM_CLASSES, 768, generator=g)
    text_feat = (class_prototypes[labels] + 0.01 * torch.randn(length, 768, generator=g)).unsqueeze(0)
    video_feat = torch.zeros(1, length, 768)
    audio_feat = torch.zeros(1, length, 768)
    speaker_ids = torch.tensor([[0, 1, 0, 1, 0, 1]])
    dialogue_mask = torch.ones(1, length, dtype=torch.bool)
    labels_batch = labels.unsqueeze(0)

    shift_label = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    shift_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, shift=True, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    shift_criterion = torch.nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        logits, shift_logits = model(video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask)
        emotion_loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), labels_batch.reshape(-1))
        valid = dialogue_mask & (shift_mask > 0.5)
        shift_loss = shift_criterion(shift_logits[valid], shift_label[valid])
        loss = emotion_loss + 0.5 * shift_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]

    final_emotion_preds = logits.argmax(dim=-1)
    assert (final_emotion_preds == labels_batch).float().mean().item() == 1.0
