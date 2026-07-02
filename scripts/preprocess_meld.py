"""Preprocess MELD.Raw into a modality-agnostic index used by both training regimes.

For each utterance clip this script:
  - extracts 16 kHz mono audio to a .wav file (ffmpeg)
  - uniformly samples 8 frames, resizes/normalizes them per MViTv2's expected
    input contract, and saves them as a single [8, 3, 224, 224] tensor (.pt)
  - records dialogue_id, utterance_id, speaker(+id), emotion label, text, and
    paths to the frame tensor / wav in a per-split index parquet, preserving
    dialogue order (sorted by dialogue_id, then utterance_id)

Clips already flagged in data/meld/bad_clips.txt (written by
scripts/download_meld.sh) are skipped. Any *additional* clips that fail to
decode during this pass are appended to the same file so both the raw and
(future) cached regimes exclude exactly the same set of utterances.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF

from rapport.data.constants import (
    EMOTION2ID,
    FRAME_MEAN,
    FRAME_SIZE,
    FRAME_STD,
    NUM_FRAMES,
    SAMPLE_RATE,
)

RESIZE_SHORT_SIDE = 256
SPLITS = ("train", "dev", "test")


def find_split_video_dir(extract_dir: Path, split: str) -> Path:
    """Locate the directory holding this split's dia{X}_utt{Y}.mp4 files."""
    candidates = sorted(extract_dir.glob(f"**/*{split}*"))
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("dia*_utt*.mp4"), None) is not None:
            return candidate
    # Fall back to a full recursive scan in case naming conventions shift.
    for mp4 in extract_dir.glob("**/dia*_utt*.mp4"):
        if split in str(mp4.parent).lower():
            return mp4.parent
    raise FileNotFoundError(f"Could not locate a video directory for split={split!r} under {extract_dir}")


def read_video_frames(mp4_path: Path) -> torch.Tensor:
    """Decodes all frames of a clip to a [T, H, W, C] uint8 tensor via PyAV."""
    container = av.open(str(mp4_path))
    try:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"no decodable frames in {mp4_path}")
    return torch.from_numpy(np.stack(frames))


def preprocess_frames(video_uint8: torch.Tensor) -> torch.Tensor:
    """[T,H,W,C] uint8 -> [NUM_FRAMES, 3, 224, 224] float, MViTv2-normalized."""
    total = video_uint8.shape[0]
    idx = torch.linspace(0, total - 1, steps=NUM_FRAMES).round().long().clamp_(0, total - 1)
    frames = video_uint8[idx].permute(0, 3, 1, 2).float() / 255.0  # [N,C,H,W]
    frames = TF.resize(frames, size=[RESIZE_SHORT_SIDE, RESIZE_SHORT_SIDE], antialias=True)
    frames = TF.center_crop(frames, output_size=[FRAME_SIZE, FRAME_SIZE])
    mean = torch.tensor(FRAME_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(FRAME_STD).view(1, 3, 1, 1)
    return (frames - mean) / std


def extract_audio(mp4_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(mp4_path),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            str(wav_path),
        ],
        check=True,
    )


def process_one(mp4_path_str: str, frame_path_str: str, wav_path_str: str) -> tuple[str, bool, str]:
    """Runs in a worker process. Returns (mp4_path, ok, error_message)."""
    mp4_path = Path(mp4_path_str)
    frame_path = Path(frame_path_str)
    wav_path = Path(wav_path_str)
    try:
        extract_audio(mp4_path, wav_path)
        video_uint8 = read_video_frames(mp4_path)
        frames = preprocess_frames(video_uint8)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(frames, frame_path)
        return mp4_path_str, True, ""
    except Exception as exc:  # noqa: BLE001 - any decode failure marks the clip bad
        wav_path.unlink(missing_ok=True)
        frame_path.unlink(missing_ok=True)
        return mp4_path_str, False, repr(exc)


def load_bad_clips(bad_clips_file: Path) -> set[str]:
    if not bad_clips_file.exists():
        return set()
    return {line.strip() for line in bad_clips_file.read_text().splitlines() if line.strip()}


def save_bad_clips(bad_clips_file: Path, bad_clips: set[str]) -> None:
    bad_clips_file.parent.mkdir(parents=True, exist_ok=True)
    bad_clips_file.write_text("\n".join(sorted(bad_clips)) + ("\n" if bad_clips else ""))


def process_split(
    split: str,
    csv_path: Path,
    video_dir: Path,
    out_dir: Path,
    bad_clips: set[str],
    workers: int,
) -> tuple[pd.DataFrame, set[str]]:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "Dialogue_ID": "dialogue_id",
            "Utterance_ID": "utterance_id",
            "Speaker": "speaker",
            "Emotion": "emotion",
            "Utterance": "text",
        }
    )

    rows = []
    tasks = []
    frames_dir = out_dir / "frames" / split
    wavs_dir = out_dir / "wavs" / split

    for row in df.itertuples(index=False):
        mp4_path = video_dir / f"dia{row.dialogue_id}_utt{row.utterance_id}.mp4"
        if not mp4_path.exists():
            bad_clips.add(str(mp4_path))
            continue
        if str(mp4_path) in bad_clips:
            continue

        frame_path = frames_dir / f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        wav_path = wavs_dir / f"dia{row.dialogue_id}_utt{row.utterance_id}.wav"
        tasks.append((str(mp4_path), str(frame_path), str(wav_path)))
        rows.append(
            {
                "dialogue_id": int(row.dialogue_id),
                "utterance_id": int(row.utterance_id),
                "speaker": str(row.speaker),
                "emotion": str(row.emotion),
                "label": EMOTION2ID[str(row.emotion)],
                "text": str(row.text),
                "frame_path": str(frame_path),
                "wav_path": str(wav_path),
                "mp4_path": str(mp4_path),
            }
        )

    ok_paths: set[str] = set()
    newly_bad: set[str] = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, *task): task[0] for task in tasks}
        done = 0
        for future in as_completed(futures):
            mp4_path_str, ok, err = future.result()
            if ok:
                ok_paths.add(mp4_path_str)
            else:
                newly_bad.add(mp4_path_str)
            done += 1
            if done % 500 == 0 or done == len(tasks):
                print(f"[preprocess_meld] {split}: {done}/{len(tasks)} processed "
                      f"({len(newly_bad)} newly bad)")

    kept_rows = [r for r in rows if r["mp4_path"] in ok_paths]
    index_df = pd.DataFrame(kept_rows).drop(columns=["mp4_path"])
    index_df = index_df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
    return index_df, newly_bad


def build_speaker_vocab(split_dfs: dict[str, pd.DataFrame]) -> dict[str, int]:
    speakers: set[str] = set()
    for df in split_dfs.values():
        speakers.update(df["speaker"].unique().tolist())
    return {speaker: i for i, speaker in enumerate(sorted(speakers))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/meld/raw/MELD.Raw", type=Path)
    parser.add_argument("--labels-dir", default="data/meld/raw/labels", type=Path)
    parser.add_argument("--out-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--bad-clips-file", default="data/meld/bad_clips.txt", type=Path)
    parser.add_argument("--speaker-vocab-file", default="data/meld/speaker_vocab.json", type=Path)
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()

    bad_clips = load_bad_clips(args.bad_clips_file)
    print(f"[preprocess_meld] starting with {len(bad_clips)} pre-flagged bad clip(s)")

    split_dfs: dict[str, pd.DataFrame] = {}
    for split in args.splits:
        csv_path = args.labels_dir / f"{split}_sent_emo.csv"
        video_dir = find_split_video_dir(args.raw_dir, split)
        print(f"[preprocess_meld] split={split} video_dir={video_dir}")

        index_df, newly_bad = process_split(split, csv_path, video_dir, args.out_dir, bad_clips, args.workers)
        bad_clips |= newly_bad
        split_dfs[split] = index_df
        print(f"[preprocess_meld] split={split}: kept {len(index_df)} utterances, "
              f"{len(newly_bad)} newly-bad clip(s) this pass")

    speaker_vocab = build_speaker_vocab(split_dfs)
    args.speaker_vocab_file.parent.mkdir(parents=True, exist_ok=True)
    args.speaker_vocab_file.write_text(json.dumps(speaker_vocab, indent=2, sort_keys=True))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, df in split_dfs.items():
        df = df.copy()
        df["speaker_id"] = df["speaker"].map(speaker_vocab)
        out_path = args.out_dir / f"{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[preprocess_meld] wrote {out_path} ({len(df)} rows)")

    save_bad_clips(args.bad_clips_file, bad_clips)
    print(f"[preprocess_meld] total bad clips (cumulative): {len(bad_clips)} -> {args.bad_clips_file}")


if __name__ == "__main__":
    main()
