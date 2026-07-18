"""Preprocess IEMOCAP into the same modality-agnostic index schema MELD uses
(docs/RECIPE.md's IEMOCAP section, Phase N5-B Step B3), so
scripts/build_feature_cache.py and every MELDCachedDataset/
MELDContextTextDataset consumer works against it unmodified.

Unlike MELD.Raw (which ships pre-cut per-utterance clips), IEMOCAP's release
is DIALOGUE-level: one wav + one avi per dialogue, with per-utterance
[start,end] timestamps and consensus emotion labels in the EmoEvaluation
.txt files. This script actually cuts the clips MELD's preprocessing gets
for free:
  - audio: ffmpeg -ss/-to cut from the dialogue wav to 16 kHz mono
  - video: 8 uniformly-sampled frames within [start,end], decoded ONCE per
    dialogue (not once per utterance -- a single PyAV streaming pass keyed
    to only the frame indices any utterance in that dialogue actually
    needs), cropped to the ACTIVE SPEAKER's half of the split-screen frame.

Split-screen convention (verified visually against 4 sample dialogues
across sessions 1/3/5, both F- and M-designated files -- see
docs/RECIPE.md): the actor wearing MoCap markers -- i.e. whichever gender
letter appears in the dialogue's own name (`Ses01F_impro01` -> F) -- is
always on the LEFT half; the other actor is on the RIGHT half. This holds
regardless of which of the two actors is speaking in a given utterance, so
the crop side is chosen by comparing the SPEAKING utterance's own gender
(parsed from its turn id, e.g. `..._M003` -> M) against the dialogue's
designation letter, not by the dialogue's letter alone.

Only 6-class-kept utterances (docs/RECIPE.md: {angry, happy, excited, sad,
neutral, frustrated}) are processed at all -- xxx/oth/fear/disgust/surprise
utterances are never cut, cached, or indexed.

Split: Session5=test, Session4=val, Sessions1-3=train (docs/RECIPE.md).
`speaker` is a persistent per-actor identity ("Ses01_F"), NOT scoped to one
dialogue-file, so the same person's turns across every dialogue in their
session map to one speaker_id and shift-label history carries correctly
within a session's worth of dialogues that share a speaker across files
(mirrors MELD's global speaker_vocab, reusing preprocess_meld.build_speaker_vocab).

Usage:
    uv run python -m scripts.preprocess_iemocap
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF

from rapport.data.constants import FRAME_MEAN, FRAME_SIZE, FRAME_STD, IEMOCAP_CODE2LABEL, IEMOCAP_EMOTION2ID, NUM_FRAMES, SAMPLE_RATE
from rapport.data.shift_labels import add_shift_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESIZE_SHORT_SIDE = 256

DEFAULT_RAW_DIR = Path.home() / "data" / "iemocap" / "iemocap" / "IEMOCAP_full_release"
SESSIONS = [f"Session{i}" for i in range(1, 6)]
SESSION_TO_SPLIT = {"Session1": "train", "Session2": "train", "Session3": "train", "Session4": "val", "Session5": "test"}

TURN_LINE_RE = re.compile(r"^\[(?P<start>[\d.]+) - (?P<end>[\d.]+)\]\t(?P<turn>\S+)\t(?P<emotion>\S+)\t")
TRANSCRIPT_LINE_RE = re.compile(r"^(?P<turn>\S+)\s+\[[\d.]+-[\d.]+\]:\s*(?P<text>.*)$")


def dialogue_designation(dialogue_name: str) -> str:
    """`Ses01F_impro01` -> 'F' (the gender letter right after the session number)."""
    m = re.match(r"Ses\d+([MF])", dialogue_name)
    assert m, f"unexpected dialogue name format: {dialogue_name!r}"
    return m.group(1)


def turn_speaker_gender(turn_id: str, dialogue_name: str) -> str:
    suffix = turn_id.split(dialogue_name, 1)[-1]
    return "F" if "_F" in suffix else "M"


def parse_emo_evaluation(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = TURN_LINE_RE.match(line)
        if m:
            rows.append(
                {"turn": m.group("turn"), "emotion": m.group("emotion"), "start": float(m.group("start")), "end": float(m.group("end"))}
            )
    return rows


def parse_transcript(path: Path) -> dict[str, str]:
    text_by_turn = {}
    if not path.exists():
        return text_by_turn
    for line in path.read_text(errors="replace").splitlines():
        m = TRANSCRIPT_LINE_RE.match(line)
        if m:
            text_by_turn[m.group("turn")] = m.group("text").strip()
    return text_by_turn


def preprocess_frame_stack(frames_uint8_list: list[np.ndarray]) -> torch.Tensor:
    """list of [H_half,W_half,3] uint8 crops (len NUM_FRAMES) -> [NUM_FRAMES,3,224,224] float, MViTv2-normalized."""
    video_uint8 = torch.from_numpy(np.stack(frames_uint8_list))  # [N,H,W,3]
    frames = video_uint8.permute(0, 3, 1, 2).float() / 255.0  # [N,3,H,W]
    frames = TF.resize(frames, size=[RESIZE_SHORT_SIDE, RESIZE_SHORT_SIDE], antialias=True)
    frames = TF.center_crop(frames, output_size=[FRAME_SIZE, FRAME_SIZE])
    mean = torch.tensor(FRAME_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(FRAME_STD).view(1, 3, 1, 1)
    return (frames - mean) / std


def extract_audio_clip(wav_path: Path, out_path: Path, start: float, end: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(wav_path),
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            str(out_path),
        ],
        check=True,
    )


def extract_video_frames_for_dialogue(avi_path: Path, utterances: list[dict], designation: str) -> dict[str, list[np.ndarray] | None]:
    """Single streaming PyAV decode pass over the dialogue's avi. For each
    utterance, computes NUM_FRAMES uniformly-sampled frame indices within
    its [start,end] window and collects only those frames (cropped to the
    active speaker's half immediately, to bound memory). If the video is
    shorter than an utterance's requested frame index (audio/video duration
    mismatch, observed directly on this release), the last successfully
    decoded frame is reused for any remaining indices -- the same
    clamp-to-valid-range spirit as MELD's `preprocess_frames`, just applied
    during a streaming pass since total frame count isn't known upfront.
    Returns {turn_id: [NUM_FRAMES x [H,W,3] uint8 crop]} or {turn_id: None}
    for utterances whose frames couldn't be extracted at all (empty video).
    """
    container = av.open(str(avi_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30000 / 1001

    needed: dict[int, list[tuple[str, int]]] = {}
    for utt in utterances:
        timestamps = np.linspace(utt["start"], utt["end"], NUM_FRAMES)
        frame_indices = [max(0, round(t * fps)) for t in timestamps]
        for pos, fi in enumerate(frame_indices):
            needed.setdefault(fi, []).append((utt["turn"], pos))
    max_needed_idx = max(needed.keys()) if needed else -1

    result: dict[str, list[np.ndarray | None]] = {utt["turn"]: [None] * NUM_FRAMES for utt in utterances}
    last_frame_arr = None
    frame_idx = 0
    try:
        for frame in container.decode(stream):
            if frame_idx in needed:
                arr = frame.to_ndarray(format="rgb24")
                last_frame_arr = arr
                for turn_id, pos in needed[frame_idx]:
                    result[turn_id][pos] = _crop_half(arr, turn_id, designation)
            frame_idx += 1
            if frame_idx > max_needed_idx:
                break
    finally:
        container.close()

    # Fill any indices beyond the actual decoded video length with the last
    # decoded frame (audio/video duration mismatch fallback, see docstring).
    if last_frame_arr is not None:
        for utt in utterances:
            turn_id = utt["turn"]
            for pos in range(NUM_FRAMES):
                if result[turn_id][pos] is None:
                    result[turn_id][pos] = _crop_half(last_frame_arr, turn_id, designation)
    else:
        for utt in utterances:
            result[utt["turn"]] = None

    return result


def _crop_half(frame_hwc: np.ndarray, turn_id: str, designation: str) -> np.ndarray:
    """Left half if the speaking turn's gender matches the dialogue's own
    designation letter (the MoCap-wearing actor, always left); right half
    otherwise. See module docstring for the visual verification."""
    _, w, _ = frame_hwc.shape
    mid = w // 2
    dialogue_name = turn_id.rsplit("_", 1)[0]
    speaker_gender = turn_speaker_gender(turn_id, dialogue_name)
    if speaker_gender == designation:
        return frame_hwc[:, :mid]
    return frame_hwc[:, mid:]


def process_dialogue(
    session: str,
    dialogue_name: str,
    raw_dir_str: str,
    out_dir_str: str,
) -> tuple[str, list[dict], str]:
    """Runs in a worker process. Returns (dialogue_name, rows, error_message)."""
    raw_dir = Path(raw_dir_str)
    out_dir = Path(out_dir_str)
    session_dir = raw_dir / session / "dialog"
    designation = dialogue_designation(dialogue_name)

    emo_path = session_dir / "EmoEvaluation" / f"{dialogue_name}.txt"
    wav_path = session_dir / "wav" / f"{dialogue_name}.wav"
    avi_path = session_dir / "avi" / "DivX" / f"{dialogue_name}.avi"
    transcript_path = session_dir / "transcriptions" / f"{dialogue_name}.txt"

    try:
        all_turns = parse_emo_evaluation(emo_path)
        kept = [t for t in all_turns if t["emotion"] in IEMOCAP_CODE2LABEL]
        kept.sort(key=lambda t: t["start"])
        if not kept:
            return dialogue_name, [], ""

        text_by_turn = parse_transcript(transcript_path)
        frames_by_turn = extract_video_frames_for_dialogue(avi_path, kept, designation)

        split = SESSION_TO_SPLIT[session]
        wavs_dir = out_dir / "wavs" / split
        frames_dir = out_dir / "frames" / split

        rows = []
        for utt in kept:
            turn_id = utt["turn"]
            gender = turn_speaker_gender(turn_id, dialogue_name)
            speaker = f"{session}_{gender}"
            frame_stack = frames_by_turn.get(turn_id)
            if frame_stack is None:
                continue

            clip_stem = f"{dialogue_name}__{turn_id}"
            wav_out = wavs_dir / f"{clip_stem}.wav"
            frame_out = frames_dir / f"{clip_stem}.pt"

            extract_audio_clip(wav_path, wav_out, utt["start"], utt["end"])
            frames_tensor = preprocess_frame_stack(frame_stack)
            frame_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(frames_tensor, frame_out)

            rows.append(
                {
                    "session": session,
                    "dialogue_name": dialogue_name,
                    "turn": turn_id,
                    "speaker": speaker,
                    "emotion": IEMOCAP_CODE2LABEL[utt["emotion"]],
                    "label": IEMOCAP_EMOTION2ID[IEMOCAP_CODE2LABEL[utt["emotion"]]],
                    "text": text_by_turn.get(turn_id, ""),
                    "frame_path": str(frame_out),
                    "wav_path": str(wav_out),
                    "start": utt["start"],
                    "end": utt["end"],
                }
            )
        return dialogue_name, rows, ""
    except Exception as exc:  # noqa: BLE001 - any decode failure marks the dialogue bad
        return dialogue_name, [], repr(exc)


def build_speaker_vocab(speakers: set[str]) -> dict[str, int]:
    return {speaker: i for i, speaker in enumerate(sorted(speakers))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, type=Path)
    parser.add_argument("--out-dir", default=PROJECT_ROOT / "data" / "iemocap" / "processed", type=Path)
    parser.add_argument("--bad-dialogues-file", default=PROJECT_ROOT / "data" / "iemocap" / "bad_dialogues.txt", type=Path)
    parser.add_argument("--speaker-vocab-file", default=PROJECT_ROOT / "data" / "iemocap" / "speaker_vocab.json", type=Path)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    args = parser.parse_args()

    dialogue_names_by_session: dict[str, list[str]] = {}
    for session in args.sessions:
        wav_dir = args.raw_dir / session / "dialog" / "wav"
        names = sorted(p.stem for p in wav_dir.glob("*.wav") if not p.name.startswith("._"))
        dialogue_names_by_session[session] = names
        print(f"[preprocess_iemocap] session={session}: {len(names)} dialogues")

    all_rows: list[dict] = []
    bad_dialogues: list[str] = []
    tasks = [
        (session, name) for session in args.sessions for name in dialogue_names_by_session[session]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_dialogue, session, name, str(args.raw_dir), str(args.out_dir)): (session, name)
            for session, name in tasks
        }
        done = 0
        for future in as_completed(futures):
            dialogue_name, rows, err = future.result()
            if err:
                bad_dialogues.append(f"{dialogue_name}: {err}")
                print(f"[preprocess_iemocap] FAILED {dialogue_name}: {err}")
            all_rows.extend(rows)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[preprocess_iemocap] {done}/{len(tasks)} dialogues processed, {len(all_rows)} utterances kept so far")

    args.bad_dialogues_file.parent.mkdir(parents=True, exist_ok=True)
    args.bad_dialogues_file.write_text("\n".join(bad_dialogues) + ("\n" if bad_dialogues else ""))

    df = pd.DataFrame(all_rows)
    speaker_vocab = build_speaker_vocab(set(df["speaker"]))
    args.speaker_vocab_file.parent.mkdir(parents=True, exist_ok=True)
    args.speaker_vocab_file.write_text(json.dumps(speaker_vocab, indent=2, sort_keys=True))
    df["speaker_id"] = df["speaker"].map(speaker_vocab)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in sorted(set(SESSION_TO_SPLIT[s] for s in args.sessions)):
        split_sessions = [s for s in args.sessions if SESSION_TO_SPLIT[s] == split]
        split_df = df[df["session"].isin(split_sessions)].copy()
        # dialogue_id: sequential per split, sorted by dialogue name (restarts
        # at 0 per split, mirroring preprocess_meld.py's convention).
        dialogue_order = {name: i for i, name in enumerate(sorted(split_df["dialogue_name"].unique()))}
        split_df["dialogue_id"] = split_df["dialogue_name"].map(dialogue_order)
        # utterance_id: positional index within the dialogue, sorted by start time.
        split_df = split_df.sort_values(["dialogue_id", "start"]).reset_index(drop=True)
        split_df["utterance_id"] = split_df.groupby("dialogue_id").cumcount()

        split_df = split_df[
            ["dialogue_id", "utterance_id", "speaker", "speaker_id", "emotion", "label", "text", "frame_path", "wav_path"]
        ]
        split_df = add_shift_labels(split_df)

        out_path = args.out_dir / f"{split}.parquet"
        split_df.to_parquet(out_path, index=False)
        print(f"[preprocess_iemocap] wrote {out_path} ({len(split_df)} rows, {split_df['dialogue_id'].nunique()} dialogues)")

    manifest = {
        "sessions_processed": args.sessions,
        "n_utterances_total": len(df),
        "n_bad_dialogues": len(bad_dialogues),
        "session_to_split": SESSION_TO_SPLIT,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[preprocess_iemocap] done. total kept utterances={len(df)}, bad dialogues={len(bad_dialogues)}")


if __name__ == "__main__":
    main()
