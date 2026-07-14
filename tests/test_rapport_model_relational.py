"""Model-level relational-memory tests (docs/SPEC_RAPPORT_COMPONENTS.md
section A9's remaining items -- pair_index/incident-only/ordering are
covered at the module level in tests/test_relational_memory.py).
"""

import torch

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


def test_relational_forward_is_finite():
    d = _dialogue(length=10, num_speakers=4, seed=0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True)
    logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
    assert logits.shape == (1, 10, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_dyadic_n2_runs_without_error():
    """A8: exactly 2 speakers (1 possible edge) must run through the same
    code path as any other N, with no special-casing required.
    """
    d = _dialogue(length=8, num_speakers=2, seed=1)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True)
    logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
    assert torch.isfinite(logits).all()


def test_a7_relational_false_reproduces_base_fusion_bit_for_bit():
    """A7: relational=False must be bit-for-bit identical to the base_fusion
    path (Step 1), on fixed input and identical weights -- this protects
    every ablation delta downstream from being an artifact of the relational
    branch accidentally touching the shared (fusion/classifier-shape-independent)
    code.
    """
    d = _dialogue(length=8, num_speakers=3, seed=2)
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=False)
    model.eval()

    with torch.no_grad():
        out_forward = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
        out_base_explicit = model._forward_base(
            d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"]
        )

    assert torch.equal(out_forward, out_base_explicit)


def test_per_dialogue_reset_two_dialogues_same_speaker_ids_dont_share_edges():
    """A2: edge state must zero-init at every dialogue start -- two different
    dialogues in the same batch that happen to reuse the same (local)
    speaker id numbering must not see each other's edge states.
    """
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True)
    model.eval()

    length = 6
    g = torch.Generator().manual_seed(0)
    video_a = torch.randn(1, length, 768, generator=g)
    audio_a = torch.randn(1, length, 768, generator=g)
    text_a = torch.randn(1, length, 768, generator=g)
    speaker_ids = torch.tensor([[0, 1, 0, 1, 0, 1]])
    dialogue_mask = torch.ones(1, length, dtype=torch.bool)

    # A completely different dialogue (different features), SAME speaker id pattern.
    g2 = torch.Generator().manual_seed(999)
    video_b = torch.randn(1, length, 768, generator=g2) * 50  # scaled, to make any leakage obvious
    audio_b = torch.randn(1, length, 768, generator=g2) * 50
    text_b = torch.randn(1, length, 768, generator=g2) * 50

    with torch.no_grad():
        out_a_alone = model(video_a, audio_a, text_a, speaker_ids, dialogue_mask)
        out_b_alone = model(video_b, audio_b, text_b, speaker_ids, dialogue_mask)

        video_batch = torch.cat([video_a, video_b], dim=0)
        audio_batch = torch.cat([audio_a, audio_b], dim=0)
        text_batch = torch.cat([text_a, text_b], dim=0)
        speaker_batch = torch.cat([speaker_ids, speaker_ids], dim=0)
        mask_batch = torch.cat([dialogue_mask, dialogue_mask], dim=0)
        out_batched = model(video_batch, audio_batch, text_batch, speaker_batch, mask_batch)

    assert torch.allclose(out_a_alone[0], out_batched[0], atol=1e-5)
    assert torch.allclose(out_b_alone[0], out_batched[1], atol=1e-5)


def test_gradient_flows_to_all_relational_and_shared_parameters():
    d = _dialogue(length=8, num_speakers=3, seed=3)
    torch.manual_seed(0)
    model = RapportModel(num_classes=NUM_CLASSES, relational=True)

    logits = model(d["video_feat"], d["audio_feat"], d["text_feat"], d["speaker_ids"], d["dialogue_mask"])
    logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient at {name}"
