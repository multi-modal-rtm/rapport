"""Cache integrity audit (diagnostics only, no training).

1. Assert cache file count per split == index parquet row count per split.
2. Assert zero cache-key overlap across splits (split-namespaced directories
   should already guarantee this structurally; verified here directly).
3. Sample N random utterances across splits, recompute V/A/T pooled features
   live through the frozen backbones, and compare against the cached
   tensors (max abs diff + allclose).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from transformers import RobertaTokenizerFast, Wav2Vec2FeatureExtractor

from rapport.models.backbones import MViTv2Backbone, RobertaBackbone, Wav2Vec2Backbone

SPLITS = ("train", "dev", "test")
MODALITIES = ("video", "audio", "text")


def check_counts_and_overlap(processed_dir: Path, cache_dir: Path) -> None:
    print("=== count check: cache files vs index rows, per split ===")
    seen_keys_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        df = pd.read_parquet(processed_dir / f"{split}.parquet")
        n_rows = len(df)
        keys = {f"dia{row.dialogue_id}_utt{row.utterance_id}" for row in df.itertuples(index=False)}
        seen_keys_by_split[split] = keys
        for modality in MODALITIES:
            n_files = len(list((cache_dir / modality / split).glob("*.pt")))
            status = "OK" if n_files == n_rows else "MISMATCH"
            print(f"  {modality}/{split}: index_rows={n_rows} cache_files={n_files} [{status}]")

    print("=== cross-split key overlap check ===")
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1 :]:
            overlap = seen_keys_by_split[split_a] & seen_keys_by_split[split_b]
            status = "OK (zero overlap)" if not overlap else f"COLLISION: {len(overlap)} shared keys"
            print(f"  {split_a} vs {split_b}: {status}")


def sample_and_recompute(processed_dir: Path, cache_dir: Path, n_samples: int, seed: int) -> None:
    print(f"=== live recompute vs cache, {n_samples} random utterances ===")
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_per_split = []
    for split in SPLITS:
        df = pd.read_parquet(processed_dir / f"{split}.parquet")
        df = df.copy()
        df["split"] = split
        frames_per_split.append(df)
    all_df = pd.concat(frames_per_split, ignore_index=True)
    sample = all_df.iloc[rng.sample(range(len(all_df)), n_samples)]

    video_bb = MViTv2Backbone().to(device)
    audio_bb = Wav2Vec2Backbone().to(device)
    text_bb = RobertaBackbone().to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    tok = RobertaTokenizerFast.from_pretrained("roberta-base")

    max_diffs = {m: 0.0 for m in MODALITIES}
    all_close = {m: True for m in MODALITIES}
    rows_checked = 0

    with torch.no_grad():
        for row in sample.itertuples(index=False):
            stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"

            frames = torch.load(row.frame_path, weights_only=True).unsqueeze(0).to(device)
            v_pooled, _ = video_bb(frames)
            v_cached = torch.load(cache_dir / "video" / row.split / stem, weights_only=True).to(device)
            v_diff = (v_pooled[0] - v_cached).abs().max().item()

            wav, _ = sf.read(row.wav_path, dtype="float32")
            ai = fe([wav], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            a_pooled, _ = audio_bb(ai["input_values"].to(device), ai["attention_mask"].to(device))
            a_cached = torch.load(cache_dir / "audio" / row.split / stem, weights_only=True).to(device)
            a_diff = (a_pooled[0] - a_cached).abs().max().item()

            ti = tok([row.text], padding=True, truncation=True, max_length=64, return_tensors="pt")
            t_pooled, _ = text_bb(ti["input_ids"].to(device), ti["attention_mask"].to(device))
            t_cached = torch.load(cache_dir / "text" / row.split / stem, weights_only=True).to(device)
            t_diff = (t_pooled[0] - t_cached).abs().max().item()

            diffs = {"video": v_diff, "audio": a_diff, "text": t_diff}
            for m, d in diffs.items():
                max_diffs[m] = max(max_diffs[m], d)
                if not torch.allclose(
                    {"video": v_pooled[0], "audio": a_pooled[0], "text": t_pooled[0]}[m],
                    {"video": v_cached, "audio": a_cached, "text": t_cached}[m],
                    atol=1e-3, rtol=1e-3,
                ):
                    all_close[m] = False
                    print(f"  ALLCLOSE FAIL split={row.split} {stem} modality={m} max_diff={d:.6f}")

            rows_checked += 1

    print(f"  checked {rows_checked} utterances")
    for m in MODALITIES:
        print(f"  {m}: max_abs_diff={max_diffs[m]:.6f} allclose(atol=1e-3,rtol=1e-3)={all_close[m]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--cache-dir", default="data/meld/cache", type=Path)
    parser.add_argument("--n-samples", default=30, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    check_counts_and_overlap(args.processed_dir, args.cache_dir)
    sample_and_recompute(args.processed_dir, args.cache_dir, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
