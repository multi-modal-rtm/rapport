"""Collate function for batching MELD dialogue samples.

Dialogues within a batch have different numbers of utterances; this pads
every per-utterance field to the batch's max dialogue length and returns a
`dialogue_mask` marking real (True) vs. padded (False) utterance slots.
Raw audio/text fields are additionally variable-length *within* an
utterance, so those get a second, inner padding pass (`audio_lengths`,
`text_attention_mask`).
"""

from __future__ import annotations

import torch


def _pad_stack(tensors: list[torch.Tensor], max_len: int, pad_value: float) -> torch.Tensor:
    """Stacks a list of [L_i, ...] tensors into [B, max_len, ...], padding along dim 0."""
    padded = []
    for t in tensors:
        pad_amount = max_len - t.shape[0]
        if pad_amount > 0:
            pad = torch.full((pad_amount, *t.shape[1:]), pad_value, dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return torch.stack(padded)


def _pad_waveforms(
    batch_waveforms: list[list[torch.Tensor]], max_dialogue_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    all_wavs = [w for dialogue in batch_waveforms for w in dialogue]
    max_audio_len = max((w.shape[0] for w in all_wavs), default=0)

    batch_size = len(batch_waveforms)
    padded = torch.zeros(batch_size, max_dialogue_len, max_audio_len)
    lengths = torch.zeros(batch_size, max_dialogue_len, dtype=torch.long)

    for i, dialogue in enumerate(batch_waveforms):
        for j, wav in enumerate(dialogue):
            padded[i, j, : wav.shape[0]] = wav
            lengths[i, j] = wav.shape[0]

    return padded, lengths


def _pad_token_ids(
    batch_ids: list[list[torch.Tensor]], max_dialogue_len: int, pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    all_ids = [ids for dialogue in batch_ids for ids in dialogue]
    max_text_len = max((ids.shape[0] for ids in all_ids), default=0)

    batch_size = len(batch_ids)
    padded = torch.full((batch_size, max_dialogue_len, max_text_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(batch_size, max_dialogue_len, max_text_len, dtype=torch.long)

    for i, dialogue in enumerate(batch_ids):
        for j, ids in enumerate(dialogue):
            padded[i, j, : ids.shape[0]] = ids
            mask[i, j, : ids.shape[0]] = 1

    return padded, mask


def collate_dialogues(batch: list[dict]) -> dict:
    """Pads a batch of dialogue samples (from MELDRawDataset or MELDCachedDataset)
    to the batch's max dialogue length, adding a boolean `dialogue_mask`.
    """
    batch_size = len(batch)
    lengths = [len(item["labels"]) for item in batch]
    max_len = max(lengths)

    dialogue_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        dialogue_mask[i, :length] = True

    out = {
        "dialogue_id": torch.tensor([item["dialogue_id"] for item in batch], dtype=torch.long),
        "dialogue_mask": dialogue_mask,
        "speaker_ids": _pad_stack([item["speaker_ids"] for item in batch], max_len, pad_value=-1),
        "labels": _pad_stack([item["labels"] for item in batch], max_len, pad_value=-1),
    }

    if "frames" in batch[0]:
        out["frames"] = _pad_stack([item["frames"] for item in batch], max_len, pad_value=0.0)

    if "video_feat" in batch[0]:
        out["video_feat"] = _pad_stack([item["video_feat"] for item in batch], max_len, pad_value=0.0)
        out["audio_feat"] = _pad_stack([item["audio_feat"] for item in batch], max_len, pad_value=0.0)
        out["text_feat"] = _pad_stack([item["text_feat"] for item in batch], max_len, pad_value=0.0)

    if "waveforms" in batch[0]:
        out["waveforms"], out["audio_lengths"] = _pad_waveforms(
            [item["waveforms"] for item in batch], max_len
        )

    if "input_ids" in batch[0]:
        out["input_ids"], out["text_attention_mask"] = _pad_token_ids(
            [item["input_ids"] for item in batch], max_len
        )
        out["text"] = [item["text"] for item in batch]

    return out
