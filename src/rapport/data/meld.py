"""MELD dialogue-level datasets for both training regimes.

Design note: each __getitem__ call returns one *whole dialogue* (its ordered
list of utterances), never a single utterance. This means a standard
`DataLoader(shuffle=True)` only ever permutes which dialogues land in which
batch — utterances within a dialogue are always read in their original,
preserved order. No custom batch sampler is needed to satisfy "batch BY
DIALOGUE, never shuffle utterances within a dialogue"; it falls out of the
dataset's indexing granularity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from rapport.data.constants import SAMPLE_RATE

DEFAULT_TOKENIZER = "bert-base-uncased"


def _group_by_dialogue(index_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Splits an index dataframe into per-dialogue frames, preserving utterance order."""
    df = index_df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    return [group.reset_index(drop=True) for _, group in df.groupby("dialogue_id", sort=True)]


class _MELDDialogueDatasetBase(Dataset):
    """Shared dialogue-grouping/index logic for the raw and cached MELD datasets."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.index_df = pd.read_parquet(self.index_path)
        self.dialogues = _group_by_dialogue(self.index_df)

    def __len__(self) -> int:
        return len(self.dialogues)


class MELDRawDataset(_MELDDialogueDatasetBase):
    """Yields raw model inputs per dialogue for on-the-fly (trainable-backbone) encoding.

    REGIME 2 (configs C, D): LoRA-adapted backbones are trainable, so features
    cannot be cached — this dataset returns the preprocessed frame tensors,
    raw audio waveforms, and tokenized text ids needed to run the backbones
    forward every step.
    """

    def __init__(
        self,
        index_path: str | Path,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        tokenizer=None,
    ):
        super().__init__(index_path)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __getitem__(self, idx: int) -> dict:
        dialogue = self.dialogues[idx]
        frames = torch.stack([torch.load(p, weights_only=True) for p in dialogue["frame_path"]])

        waveforms = []
        for wav_path in dialogue["wav_path"]:
            wav, sr = sf.read(wav_path, dtype="float32")
            assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr} Hz for {wav_path}"
            waveforms.append(torch.from_numpy(wav))

        encoded = self.tokenizer(list(dialogue["text"]), padding=False, truncation=True)

        return {
            "dialogue_id": int(dialogue["dialogue_id"].iloc[0]),
            "speaker_ids": torch.tensor(dialogue["speaker_id"].tolist(), dtype=torch.long),
            "labels": torch.tensor(dialogue["label"].tolist(), dtype=torch.long),
            "frames": frames,  # [L, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE]
            "waveforms": waveforms,  # list[Tensor], length L, variable-length raw audio
            "input_ids": [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]],
            "text": list(dialogue["text"]),
        }


class MELDCachedDataset(_MELDDialogueDatasetBase):
    """Yields precomputed 768-d V/A/T feature vectors per utterance.

    REGIME 1 (configs A, B): backbones are frozen, so features can be
    extracted once (Phase 3, after the frozen backbones exist) and cached to
    disk under cache_dir/{video,audio,text}/{split}/dia{d}_utt{u}.pt for fast
    epochs. The split subdirectory is required: dialogue_id restarts from 0 in
    each MELD split, so a flat layout silently collides utterances across
    train/dev/test.
    """

    FEATURE_DIM = 768

    def __init__(
        self,
        index_path: str | Path,
        cache_dir: str | Path,
        text_cache_subdir: str = "text",
        load_av_tokens: bool = False,
        load_text_logits: bool = False,
        text_feature_dim: int | None = None,
    ):
        """`text_cache_subdir` selects which cached text representation to load
        for the "text" modality -- "text" (default) is the frozen, non-contextual
        RoBERTa cache used by `speaker_only`; "text_ctx" is Phase T's frozen
        contextual encoder cache (docs/PHASE_N4.md Step 0), used by the RAPPORT
        fusion configs (base_fusion, full, minus_*). Video/audio always come
        from their own frozen caches regardless of this setting.

        `load_av_tokens=True` additionally loads the pre-pooling A/V token
        sequences (cache_dir/{video,audio}_tokens/...) for Phase N4's
        temporal-attention config (docs/SPEC_RAPPORT_COMPONENTS.md section
        C) -- video tokens are fixed-length (392, no padding needed within
        an utterance); audio tokens are variable-length and padded per
        utterance in `collate_dialogues`.

        `load_text_logits=True` additionally loads the frozen Phase T text
        classifier's own 7-d logits (cache_dir/{text_cache_subdir}_logits/...)
        for Phase N4-R's residual redesign (spec v1.1, docs/PHASE_N4R.md).

        `text_feature_dim` (default None -> FEATURE_DIM, i.e. 768, unchanged
        behavior) overrides the expected width of the text_feat cache only --
        video/audio stay validated at FEATURE_DIM regardless. Second-encoder
        replication (docs/PREREG_DeBERTa-v3-large.md) passes 1024 here for
        DeBERTa-v3-large's text caches.
        """
        super().__init__(index_path)
        self.cache_dir = Path(cache_dir)
        self.split = Path(index_path).stem
        self.text_cache_subdir = text_cache_subdir
        self.load_av_tokens = load_av_tokens
        self.load_text_logits = load_text_logits
        self.text_feature_dim = self.FEATURE_DIM if text_feature_dim is None else text_feature_dim

    def _load_feature(self, modality: str, dialogue_id: int, utterance_id: int, expected_dim: int | None = None) -> torch.Tensor:
        expected_dim = self.FEATURE_DIM if expected_dim is None else expected_dim
        path = self.cache_dir / modality / self.split / f"dia{dialogue_id}_utt{utterance_id}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached {modality} feature at {path}. MELDCachedDataset requires "
                "the Phase 3 feature-extraction pass (frozen backbones) to have run first."
            )
        feat = torch.load(path, weights_only=True)
        if tuple(feat.shape) != (expected_dim,):
            raise ValueError(f"expected shape ({expected_dim},), got {tuple(feat.shape)} at {path}")
        return feat

    def _load_raw(self, modality: str, dialogue_id: int, utterance_id: int) -> torch.Tensor:
        """Loads a cached tensor with no shape assertion at all (e.g. the
        num_classes-d text_logits cache -- see `load_text_logits`)."""
        path = self.cache_dir / modality / self.split / f"dia{dialogue_id}_utt{utterance_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached {modality} tensor at {path}.")
        return torch.load(path, weights_only=True)

    def _load_tokens(self, modality: str, dialogue_id: int, utterance_id: int) -> torch.Tensor:
        """Loads a variable-length [T, FEATURE_DIM] pre-pooling token sequence
        (no shape assertion beyond the feature dim -- T varies by utterance)."""
        path = self.cache_dir / modality / self.split / f"dia{dialogue_id}_utt{utterance_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached {modality} tokens at {path}.")
        tokens = torch.load(path, weights_only=True)
        if tokens.shape[-1] != self.FEATURE_DIM:
            raise ValueError(f"expected last dim {self.FEATURE_DIM}, got {tuple(tokens.shape)} at {path}")
        return tokens

    def __getitem__(self, idx: int) -> dict:
        dialogue = self.dialogues[idx]
        video_feat, audio_feat, text_feat = [], [], []
        video_tokens, audio_tokens = [], []
        text_logits = []
        logits_subdir = f"{self.text_cache_subdir}_logits"
        for row in dialogue.itertuples(index=False):
            video_feat.append(self._load_feature("video", row.dialogue_id, row.utterance_id))
            audio_feat.append(self._load_feature("audio", row.dialogue_id, row.utterance_id))
            text_feat.append(
                self._load_feature(
                    self.text_cache_subdir, row.dialogue_id, row.utterance_id, expected_dim=self.text_feature_dim
                )
            )
            if self.load_av_tokens:
                video_tokens.append(self._load_tokens("video_tokens", row.dialogue_id, row.utterance_id))
                audio_tokens.append(self._load_tokens("audio_tokens", row.dialogue_id, row.utterance_id))
            if self.load_text_logits:
                text_logits.append(self._load_raw(logits_subdir, row.dialogue_id, row.utterance_id))

        item = {
            "dialogue_id": int(dialogue["dialogue_id"].iloc[0]),
            "speaker_ids": torch.tensor(dialogue["speaker_id"].tolist(), dtype=torch.long),
            "labels": torch.tensor(dialogue["label"].tolist(), dtype=torch.long),
            "video_feat": torch.stack(video_feat),
            "audio_feat": torch.stack(audio_feat),
            "text_feat": torch.stack(text_feat),
        }
        if self.load_av_tokens:
            item["video_tokens"] = video_tokens  # list[Tensor[392, 768]], length L (fixed length, still a list for collate uniformity)
            item["audio_tokens"] = audio_tokens  # list[Tensor[T_i, 768]], length L, variable T_i
        if self.load_text_logits:
            item["text_logits"] = torch.stack(text_logits)  # [L, num_classes]

        if "shift_label" in dialogue.columns:
            item["shift_label"] = torch.tensor(dialogue["shift_label"].tolist(), dtype=torch.float32)
            item["shift_mask"] = torch.tensor(dialogue["shift_mask"].tolist(), dtype=torch.float32)

        return item
