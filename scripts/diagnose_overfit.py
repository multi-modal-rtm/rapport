"""Tiny-overfit diagnostic for the speaker_only SocialGNN (diagnostics only, no training-code changes).

Trains on 5 real cached dialogues for 300 steps — this MUST reach ~100%
train accuracy for any reasonably-expressive model; failure implicates
model/collate/masking/state-update code, independent of feature quality.

Also directly probes:
  - padding exclusion from loss and from the accuracy metric
  - graph-state isolation across dialogues within a batch (no cross-talk
    between batch elements that happen to share a speaker_id)
"""

from __future__ import annotations

import copy

import torch
from torch.optim import AdamW

from rapport.data import MELDCachedDataset, collate_dialogues
from rapport.data.constants import EMOTION_LABELS
from rapport.models.social_gnn import SocialGNN
from rapport.training.losses import FocalLoss

NUM_CLASSES = len(EMOTION_LABELS)


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def tiny_overfit_test(batch: dict, device: torch.device, steps: int = 300, lr: float = 3e-3) -> None:
    print(f"=== tiny-overfit test: 5 dialogues, {steps} steps, lr={lr} ===")
    torch.manual_seed(0)
    model = SocialGNN(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = FocalLoss(gamma=3.0, ignore_index=-1)

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        logits = model(
            batch["video_feat"], batch["audio_feat"], batch["text_feat"],
            batch["speaker_ids"], batch["dialogue_mask"],
        )
        loss = criterion(logits.reshape(-1, NUM_CLASSES), batch["labels"].reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            acc = masked_accuracy(logits, batch["labels"], batch["dialogue_mask"])
            print(f"  step {step:03d}: loss={loss.item():.4f} train_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(
            batch["video_feat"], batch["audio_feat"], batch["text_feat"],
            batch["speaker_ids"], batch["dialogue_mask"],
        )
    final_acc = masked_accuracy(logits, batch["labels"], batch["dialogue_mask"])
    n_utterances = batch["dialogue_mask"].sum().item()
    print(f"  FINAL train accuracy: {final_acc:.4f} over {n_utterances} unmasked utterances")
    print(f"  verdict: {'PASS (~100%)' if final_acc >= 0.95 else 'FAIL — model/collate/masking/state bug likely'}")
    return model


def padding_leak_check(batch: dict, model: SocialGNN, device: torch.device) -> None:
    print("\n=== padding leak check (loss and accuracy) ===")
    criterion = FocalLoss(gamma=3.0, ignore_index=-1)
    mask = batch["dialogue_mask"]
    n_padded = (~mask).sum().item()
    if n_padded == 0:
        print("  no padded positions in this batch — cannot test, skipping")
        return

    with torch.no_grad():
        logits_orig = model(
            batch["video_feat"], batch["audio_feat"], batch["text_feat"],
            batch["speaker_ids"], batch["dialogue_mask"],
        )
        loss_orig = criterion(logits_orig.reshape(-1, NUM_CLASSES), batch["labels"].reshape(-1))
        acc_orig = masked_accuracy(logits_orig, batch["labels"], mask)

    # Corrupt padded positions: labels to an out-of-range-safe wrong class, and
    # features to random noise. If padding leaks into loss/metrics, this changes them.
    corrupted = copy.deepcopy(batch)
    pad_positions = ~mask
    corrupted["labels"] = corrupted["labels"].clone()
    corrupted["labels"][pad_positions] = 0  # was -1 (ignore sentinel); now a real-looking wrong class
    for key in ("video_feat", "audio_feat", "text_feat"):
        corrupted[key] = corrupted[key].clone()
        corrupted[key][pad_positions] = torch.randn_like(corrupted[key][pad_positions]) * 100

    with torch.no_grad():
        logits_corrupt = model(
            corrupted["video_feat"], corrupted["audio_feat"], corrupted["text_feat"],
            corrupted["speaker_ids"], corrupted["dialogue_mask"],
        )
        # loss computed with ORIGINAL labels (mask unchanged), only features at padded spots were corrupted —
        # if the model output at REAL positions changes, that would mean padding leaks into the computation.
        loss_feat_corrupt = criterion(logits_corrupt.reshape(-1, NUM_CLASSES), batch["labels"].reshape(-1))
        acc_feat_corrupt = masked_accuracy(logits_corrupt, batch["labels"], mask)

        # loss computed with CORRUPTED labels at padded spots (features unchanged) —
        # if ignore_index masking works, this must exactly equal the original loss.
        loss_label_corrupt = criterion(logits_orig.reshape(-1, NUM_CLASSES), corrupted["labels"].reshape(-1))

    print(f"  padded positions in batch: {n_padded}")
    print(f"  loss (orig)                         = {loss_orig.item():.6f}")
    print(f"  loss (padded LABELS corrupted)       = {loss_label_corrupt.item():.6f}  "
          f"[{'OK: unchanged' if abs(loss_orig.item() - loss_label_corrupt.item()) < 1e-6 else 'LEAK: loss changed!'}]")
    print(f"  logits at real positions (padded FEATURES corrupted) match orig: "
          f"{torch.allclose(logits_orig[mask], logits_corrupt[mask], atol=1e-4)}  "
          f"[{'OK: no leak into real positions' if torch.allclose(logits_orig[mask], logits_corrupt[mask], atol=1e-4) else 'LEAK: real-position outputs changed!'}]")
    print(f"  accuracy (orig) = {acc_orig:.4f}, accuracy (padded features corrupted) = {acc_feat_corrupt:.4f}  "
          f"[{'OK: unchanged' if abs(acc_orig - acc_feat_corrupt) < 1e-6 else 'LEAK: accuracy changed!'}]")


def state_isolation_check(device: torch.device) -> None:
    print("\n=== graph-state isolation check (shared speaker_id across batch elements) ===")
    torch.manual_seed(1)
    model = SocialGNN(num_classes=NUM_CLASSES, dropout=0.0).to(device)  # dropout=0 for a clean deterministic check
    model.eval()

    L = 4
    # Two dialogues, both using speaker_id 0 for every utterance (deliberately colliding
    # speaker ids across batch elements), but with completely different feature content.
    video_a = torch.randn(1, L, 768, device=device)
    audio_a = torch.randn(1, L, 768, device=device)
    text_a = torch.randn(1, L, 768, device=device)
    speaker_a = torch.zeros(1, L, dtype=torch.long, device=device)
    mask_a = torch.ones(1, L, dtype=torch.bool, device=device)

    video_b = torch.randn(1, L, 768, device=device) * 50  # very different scale/content
    audio_b = torch.randn(1, L, 768, device=device) * 50
    text_b = torch.randn(1, L, 768, device=device) * 50
    speaker_b = torch.zeros(1, L, dtype=torch.long, device=device)
    mask_b = torch.ones(1, L, dtype=torch.bool, device=device)

    with torch.no_grad():
        logits_a_alone = model(video_a, audio_a, text_a, speaker_a, mask_a)
        logits_b_alone = model(video_b, audio_b, text_b, speaker_b, mask_b)

        video_batch = torch.cat([video_a, video_b], dim=0)
        audio_batch = torch.cat([audio_a, audio_b], dim=0)
        text_batch = torch.cat([text_a, text_b], dim=0)
        speaker_batch = torch.cat([speaker_a, speaker_b], dim=0)
        mask_batch = torch.cat([mask_a, mask_b], dim=0)
        logits_batched = model(video_batch, audio_batch, text_batch, speaker_batch, mask_batch)

    a_matches = torch.allclose(logits_a_alone[0], logits_batched[0], atol=1e-5)
    b_matches = torch.allclose(logits_b_alone[0], logits_batched[1], atol=1e-5)
    print(f"  dialogue A (alone) matches dialogue A (batched with B, both speaker_id=0): {a_matches}")
    print(f"  dialogue B (alone) matches dialogue B (batched with A, both speaker_id=0): {b_matches}")
    print(f"  verdict: {'OK: state is isolated per batch element' if a_matches and b_matches else 'LEAK: state crosses batch elements!'}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MELDCachedDataset("data/meld/processed/train.parquet", "data/meld/cache")

    torch.manual_seed(0)
    indices = list(range(5))
    items = [dataset[i] for i in indices]
    lengths = [len(item["labels"]) for item in items]
    print(f"picked dialogues (indices {indices}), lengths={lengths}")

    batch = collate_dialogues(items)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    model = tiny_overfit_test(batch, device)
    padding_leak_check(batch, model, device)
    state_isolation_check(device)


if __name__ == "__main__":
    main()
