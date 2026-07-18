"""Structural tests for the IEMOCAP index, mirroring tests/test_meld_data.py.

MELDRawDataset/MELDCachedDataset are schema-generic (frame_path/wav_path
point at already-preprocessed tensors/wavs, nothing MELD-specific in the
loading code) so they're reused here unmodified against
data/iemocap/processed instead of data/meld/processed.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from rapport.data import MELDCachedDataset, MELDRawDataset, collate_dialogues
from rapport.data.constants import FRAME_SIZE, IEMOCAP_EMOTION_LABELS, NUM_FRAMES, SAMPLE_RATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "iemocap" / "processed"
SPEAKER_VOCAB_FILE = PROJECT_ROOT / "data" / "iemocap" / "speaker_vocab.json"

TRAIN_INDEX = PROCESSED_DIR / "train.parquet"
VAL_INDEX = PROCESSED_DIR / "val.parquet"
TEST_INDEX = PROCESSED_DIR / "test.parquet"

# Cross-validated against docs/iemocap_inventory.md's independent per-session
# 6-class audit (Step B1) -- Session1+2+3=train, Session4=val, Session5=test.
PUBLISHED_SPLIT_COUNTS = {"train": 1365 + 1348 + 1533, "val": 1512, "test": 1622}

requires_processed_data = pytest.mark.skipif(
    not TRAIN_INDEX.exists(), reason="run scripts/preprocess_iemocap.py first"
)


@requires_processed_data
def test_split_counts_match_inventory_audit():
    for split, index_path in (("train", TRAIN_INDEX), ("val", VAL_INDEX), ("test", TEST_INDEX)):
        df = pd.read_parquet(index_path)
        assert len(df) == PUBLISHED_SPLIT_COUNTS[split], (
            f"{split}: expected {PUBLISHED_SPLIT_COUNTS[split]} utterances (docs/iemocap_inventory.md "
            f"6-class audit), got {len(df)}"
        )


@requires_processed_data
def test_all_six_labels_present_in_every_split():
    for index_path in (TRAIN_INDEX, VAL_INDEX, TEST_INDEX):
        df = pd.read_parquet(index_path)
        assert set(df["emotion"].unique()) == set(IEMOCAP_EMOTION_LABELS)
        assert set(df["label"].unique()) == set(range(len(IEMOCAP_EMOTION_LABELS)))


@requires_processed_data
def test_dialogue_ordering_and_speaker_consistency():
    dataset = MELDRawDataset(TRAIN_INDEX, tokenizer_name="bert-base-uncased")
    speaker_vocab = json.loads(SPEAKER_VOCAB_FILE.read_text())

    for idx in range(min(20, len(dataset))):
        dialogue_df = dataset.dialogues[idx]

        utt_ids = dialogue_df["utterance_id"].tolist()
        assert utt_ids == sorted(utt_ids), f"dialogue {idx} utterance order not preserved: {utt_ids}"

        # Dyadic: exactly one male + one female persistent speaker per dialogue.
        speakers_seen = set(dialogue_df["speaker"])
        assert len(speakers_seen) == 2, f"dialogue {idx} is not dyadic: speakers={speakers_seen}"

        for speaker, speaker_id in zip(dialogue_df["speaker"], dialogue_df["speaker_id"]):
            assert speaker_vocab[speaker] == speaker_id

        name_to_id = {}
        for speaker, speaker_id in zip(dialogue_df["speaker"], dialogue_df["speaker_id"]):
            if speaker in name_to_id:
                assert name_to_id[speaker] == speaker_id
            else:
                name_to_id[speaker] = speaker_id


@requires_processed_data
def test_speaker_vocab_has_ten_speakers():
    speaker_vocab = json.loads(SPEAKER_VOCAB_FILE.read_text())
    assert len(speaker_vocab) == 10, f"expected 10 (session,gender) speaker slots, got {len(speaker_vocab)}"


@requires_processed_data
def test_frame_and_audio_shapes():
    dataset = MELDRawDataset(TRAIN_INDEX, tokenizer_name="bert-base-uncased")
    item = dataset[0]

    L = len(item["labels"])
    assert item["frames"].shape == (L, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE)

    import soundfile as sf

    for wav_path, waveform in zip(dataset.dialogues[0]["wav_path"], item["waveforms"]):
        info = sf.info(wav_path)
        assert info.samplerate == SAMPLE_RATE
        assert info.channels == 1
        assert waveform.ndim == 1


@requires_processed_data
def test_raw_dataset_collate_pads_and_masks():
    dataset = MELDRawDataset(TRAIN_INDEX, tokenizer_name="bert-base-uncased")
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    out = collate_dialogues(batch)

    batch_size = len(batch)
    max_len = max(len(item["labels"]) for item in batch)

    assert out["labels"].shape == (batch_size, max_len)
    assert out["dialogue_mask"].shape == (batch_size, max_len)
    assert out["frames"].shape == (batch_size, max_len, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE)

    for i, item in enumerate(batch):
        length = len(item["labels"])
        assert out["dialogue_mask"][i, :length].all()
        if length < max_len:
            assert not out["dialogue_mask"][i, length:].any()


def test_cached_dataset_and_collate_with_synthetic_iemocap_features(tmp_path):
    """Same structural check as MELD's synthetic-cache test, with IEMOCAP's
    6-class labels/speaker-naming convention -- doesn't require the real
    corpus."""
    index_df = pd.DataFrame(
        {
            "dialogue_id": [0, 0, 0, 1, 1],
            "utterance_id": [0, 1, 2, 0, 1],
            "speaker": ["Session1_F", "Session1_M", "Session1_F", "Session2_F", "Session2_M"],
            "speaker_id": [0, 1, 0, 2, 3],
            "emotion": ["neutral", "angry", "frustrated", "sad", "happy"],
            "label": [4, 0, 5, 3, 1],
            "text": ["hi", "no", "ugh", "sigh", "great"],
        }
    )
    index_path = tmp_path / "train.parquet"
    index_df.to_parquet(index_path, index=False)

    cache_dir = tmp_path / "cache"
    for modality in ("video", "audio", "text"):
        (cache_dir / modality / "train").mkdir(parents=True)
    for _, row in index_df.iterrows():
        for modality in ("video", "audio", "text"):
            path = cache_dir / modality / "train" / f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
            torch.save(torch.randn(768), path)

    dataset = MELDCachedDataset(index_path, cache_dir)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0["video_feat"].shape == (3, 768)
    assert item0["speaker_ids"].tolist() == [0, 1, 0]

    batch = [dataset[0], dataset[1]]
    out = collate_dialogues(batch)
    assert out["video_feat"].shape == (2, 3, 768)
    assert out["dialogue_mask"].tolist() == [[True, True, True], [True, True, False]]
