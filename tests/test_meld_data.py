import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import torch

from rapport.data import MELDCachedDataset, MELDRawDataset, collate_dialogues
from rapport.data.constants import EMOTION_LABELS, FRAME_SIZE, NUM_FRAMES, SAMPLE_RATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "meld" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "meld" / "raw" / "labels"
BAD_CLIPS_FILE = PROJECT_ROOT / "data" / "meld" / "bad_clips.txt"

TRAIN_INDEX = PROCESSED_DIR / "train.parquet"

PUBLISHED_TRAIN_COUNTS = {
    "neutral": 4710,
    "joy": 1743,
    "surprise": 1205,
    "anger": 1109,
    "sadness": 683,
    "fear": 268,
    "disgust": 271,
}

CLIP_NAME_RE = re.compile(r"dia(\d+)_utt(\d+)\.mp4$")

requires_processed_data = pytest.mark.skipif(
    not TRAIN_INDEX.exists(), reason="run scripts/preprocess_meld.py first"
)


def _bad_clip_emotion_deltas(split: str) -> Counter:
    """For every bad-clip path belonging to `split`, look up its published emotion label."""
    if not BAD_CLIPS_FILE.exists():
        return Counter()
    csv_path = LABELS_DIR / f"{split}_sent_emo.csv"
    df = pd.read_csv(csv_path)
    label_by_key = {
        (int(row.Dialogue_ID), int(row.Utterance_ID)): row.Emotion for row in df.itertuples(index=False)
    }

    deltas: Counter = Counter()
    for line in BAD_CLIPS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or split not in line:
            continue
        match = CLIP_NAME_RE.search(line)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in label_by_key:
            deltas[label_by_key[key]] += 1
    return deltas


@requires_processed_data
def test_train_class_distribution_matches_published_minus_bad_clips():
    df = pd.read_parquet(TRAIN_INDEX)
    actual_counts = df["emotion"].value_counts().to_dict()

    excluded = _bad_clip_emotion_deltas("train")
    expected_counts = {
        emotion: count - excluded.get(emotion, 0) for emotion, count in PUBLISHED_TRAIN_COUNTS.items()
    }

    for emotion, expected in expected_counts.items():
        assert actual_counts.get(emotion, 0) == expected, (
            f"{emotion}: expected {expected} (published {PUBLISHED_TRAIN_COUNTS[emotion]} "
            f"- {excluded.get(emotion, 0)} excluded), got {actual_counts.get(emotion, 0)}"
        )


@requires_processed_data
def test_dialogue_ordering_and_speaker_consistency():
    dataset = MELDRawDataset(TRAIN_INDEX, tokenizer_name="bert-base-uncased")
    speaker_vocab = json.loads((PROJECT_ROOT / "data" / "meld" / "speaker_vocab.json").read_text())

    # Check a sample of dialogues (full dataset is slow to tokenize one-by-one in a test).
    for idx in range(min(20, len(dataset))):
        dialogue_df = dataset.dialogues[idx]

        utt_ids = dialogue_df["utterance_id"].tolist()
        assert utt_ids == sorted(utt_ids), f"dialogue {idx} utterance order not preserved: {utt_ids}"

        for speaker, speaker_id in zip(dialogue_df["speaker"], dialogue_df["speaker_id"]):
            assert speaker_vocab[speaker] == speaker_id

        # Same speaker name within a dialogue must always map to the same id.
        name_to_id = {}
        for speaker, speaker_id in zip(dialogue_df["speaker"], dialogue_df["speaker_id"]):
            if speaker in name_to_id:
                assert name_to_id[speaker] == speaker_id
            else:
                name_to_id[speaker] = speaker_id


@requires_processed_data
def test_frame_and_audio_shapes():
    dataset = MELDRawDataset(TRAIN_INDEX, tokenizer_name="bert-base-uncased")
    item = dataset[0]

    L = len(item["labels"])
    assert item["frames"].shape == (L, NUM_FRAMES, 3, FRAME_SIZE, FRAME_SIZE)

    for wav_path, waveform in zip(dataset.dialogues[0]["wav_path"], item["waveforms"]):
        import soundfile as sf

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


def test_cached_dataset_and_collate_with_synthetic_features(tmp_path):
    """Structural test for MELDCachedDataset — doesn't require Phase 3 (real
    frozen-backbone features) to exist yet; builds a tiny synthetic cache instead.
    """
    index_df = pd.DataFrame(
        {
            "dialogue_id": [0, 0, 0, 1, 1],
            "utterance_id": [0, 1, 2, 0, 1],
            "speaker": ["A", "B", "A", "C", "A"],
            "speaker_id": [0, 1, 0, 2, 0],
            "emotion": ["neutral", "joy", "anger", "sadness", "neutral"],
            "label": [0, 1, 3, 2, 0],
            "text": ["hi", "hello", "grr", "sigh", "ok"],
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
    assert len(dataset) == 2  # two dialogues

    item0 = dataset[0]
    assert item0["video_feat"].shape == (3, 768)
    assert item0["audio_feat"].shape == (3, 768)
    assert item0["text_feat"].shape == (3, 768)
    assert item0["speaker_ids"].tolist() == [0, 1, 0]

    batch = [dataset[0], dataset[1]]
    out = collate_dialogues(batch)
    assert out["video_feat"].shape == (2, 3, 768)
    assert out["dialogue_mask"].tolist() == [[True, True, True], [True, True, False]]
