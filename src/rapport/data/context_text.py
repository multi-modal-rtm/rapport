"""Context-window text input construction for the Phase T contextual text encoder.

For utterance t, builds "<spk_{t-k}>: u_{t-k} </s> ... <spk_{t-1}>: u_{t-1}
</s> <spk_t>: u_t" using the k most recent PRECEDING utterances in the same
dialogue (never future utterances) -- speaker names as plain-text prefixes
(e.g. "Joey:"), joined with the tokenizer's own separator token so it's
encoded as a real special-token id, not literal characters. If the full
window doesn't fit in max_length tokens, context segments are dropped
OLDEST-first until it does; the current utterance (the last segment) is
always kept intact and is never itself a candidate for dropping.

Batching here is utterance-level, not dialogue-level: this phase has no
recurrent/graph state across utterances (see docs/RECIPE.md, Phase T), so
`MELDContextTextDataset` flattens every dialogue's utterances into one long
list of independent, randomly-shufflable training examples -- unlike
`rapport.data.meld`'s dialogue-grouped datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_K = 8
DEFAULT_MAX_LENGTH = 256


def context_window(
    speakers: list[str], utterances: list[str], t: int, k: int = DEFAULT_K
) -> list[tuple[str, str]]:
    """Returns the ordered (speaker, text) segments for utterance t: up to k
    preceding utterances (oldest first) followed by utterance t itself.

    Never includes an index > t, so this can never leak future context. For
    t == 0 (the first utterance of a dialogue), returns a single segment --
    utterance 0 alone, with no preceding context.
    """
    if not (0 <= t < len(speakers)):
        raise IndexError(f"t={t} out of range for dialogue of length {len(speakers)}")
    start = max(0, t - k)
    return list(zip(speakers[start : t + 1], utterances[start : t + 1]))


def encode_with_context(
    tokenizer,
    speakers: list[str],
    utterances: list[str],
    t: int,
    k: int = DEFAULT_K,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, torch.Tensor]:
    """Tokenizes utterance t's context window, dropping the OLDEST context
    segment first (repeatedly) if the encoded length exceeds max_length,
    always preserving the current utterance (the last segment) whole.

    Deterministic: depends only on the inputs, tokenizer, k, and max_length --
    no randomness anywhere in this path.
    """
    segments = context_window(speakers, utterances, t, k)
    sep = tokenizer.sep_token

    input_ids: list[int] = []
    while segments:
        text = f" {sep} ".join(f"{speaker}: {utterance}" for speaker, utterance in segments)
        input_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(input_ids) <= max_length or len(segments) == 1:
            break
        segments = segments[1:]  # drop the oldest remaining context segment

    if len(input_ids) > max_length:
        # Only the current utterance is left and it alone still overflows --
        # last-resort right-truncation (the current utterance itself must be
        # cut; there is nothing left to drop instead).
        input_ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


class MELDContextTextDataset(Dataset):
    """Flat, utterance-level MELD dataset: each item is one utterance plus its
    preceding-context window (`encode_with_context`), for the LoRA-tuned
    contextual text classifier. No dialogue grouping at the batch level --
    see module docstring.
    """

    def __init__(
        self,
        index_path: str | Path,
        tokenizer,
        k: int = DEFAULT_K,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        df = pd.read_parquet(index_path).sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
        self.index_df = df  # exposed for label-distribution use (e.g. compute_class_priors), matches MELDCachedDataset
        self.dialogues = {
            dialogue_id: group.reset_index(drop=True)
            for dialogue_id, group in df.groupby("dialogue_id", sort=True)
        }
        self.index: list[tuple[int, int]] = [
            (dialogue_id, t) for dialogue_id, group in self.dialogues.items() for t in range(len(group))
        ]
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        dialogue_id, t = self.index[idx]
        group = self.dialogues[dialogue_id]
        item = encode_with_context(
            self.tokenizer, group["speaker"].tolist(), group["text"].tolist(), t, self.k, self.max_length
        )
        item["label"] = torch.tensor(int(group["label"].iloc[t]), dtype=torch.long)
        return item


class ContextTextCollator:
    """Pads a batch of `encode_with_context` outputs to the batch's max length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.empty(batch_size, dtype=torch.long)

        for i, item in enumerate(batch):
            length = item["input_ids"].shape[0]
            input_ids[i, :length] = item["input_ids"]
            attention_mask[i, :length] = item["attention_mask"]
            labels[i] = item["label"]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
