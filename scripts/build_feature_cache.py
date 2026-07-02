"""Runs the frozen v1 backbones (MViTv2, Wav2Vec 2.0, RoBERTa) once over every
preprocessed MELD utterance and caches pooled 768-d V/A/T vectors plus their
pre-pooling token sequences to data/meld/cache/. This serves configs A and B
(frozen-backbone regime) via MELDCachedDataset.

Audio/text token sequences are variable-length; audio is additionally strided
and capped (AUDIO_TOKEN_STRIDE, AUDIO_TOKEN_MAX_LEN) since wav2vec2's raw
feature rate is much finer-grained than needed for later temporal attention.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from transformers import RobertaTokenizerFast, Wav2Vec2FeatureExtractor

from socialarcnet.data.cache_constants import AUDIO_TOKEN_MAX_LEN, AUDIO_TOKEN_STRIDE, TEXT_TOKEN_MAX_LEN
from socialarcnet.models.backbones import MViTv2Backbone, RobertaBackbone, Wav2Vec2Backbone

SPLITS = ("train", "dev", "test")
CACHE_SUBDIRS = ("video", "audio", "text", "video_tokens", "audio_tokens", "text_tokens")


def _is_cached(out_dir: Path, split: str, dialogue_id: int, utterance_id: int) -> bool:
    stem = f"dia{dialogue_id}_utt{utterance_id}.pt"
    return all((out_dir / subdir / split / stem).exists() for subdir in CACHE_SUBDIRS)


@torch.no_grad()
def process_batch(
    batch: pd.DataFrame,
    split: str,
    video_backbone: MViTv2Backbone,
    audio_backbone: Wav2Vec2Backbone,
    text_backbone: RobertaBackbone,
    audio_feature_extractor: Wav2Vec2FeatureExtractor,
    tokenizer: RobertaTokenizerFast,
    device: torch.device,
    out_dir: Path,
) -> None:
    # --- video ---
    frames = torch.stack([torch.load(p, weights_only=True) for p in batch["frame_path"]]).to(device)
    v_pooled, v_tokens = video_backbone(frames)

    # --- audio ---
    waveforms = [sf.read(p, dtype="float32")[0] for p in batch["wav_path"]]
    audio_inputs = audio_feature_extractor(
        waveforms, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True
    )
    audio_attention_mask = audio_inputs["attention_mask"].to(device)
    a_pooled, a_tokens = audio_backbone(audio_inputs["input_values"].to(device), audio_attention_mask)
    a_feat_mask = audio_backbone.model._get_feature_vector_attention_mask(a_tokens.shape[1], audio_attention_mask)

    # --- text ---
    text_inputs = tokenizer(
        list(batch["text"]), padding=True, truncation=True, max_length=TEXT_TOKEN_MAX_LEN, return_tensors="pt"
    )
    text_attention_mask = text_inputs["attention_mask"].to(device)
    t_pooled, t_tokens = text_backbone(text_inputs["input_ids"].to(device), text_attention_mask)

    for i, row in enumerate(batch.itertuples(index=False)):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"

        torch.save(v_pooled[i].float().cpu(), out_dir / "video" / split / stem)
        torch.save(a_pooled[i].float().cpu(), out_dir / "audio" / split / stem)
        torch.save(t_pooled[i].float().cpu(), out_dir / "text" / split / stem)

        torch.save(v_tokens[i].float().cpu(), out_dir / "video_tokens" / split / stem)

        a_len = int(a_feat_mask[i].sum().item())
        a_seq = a_tokens[i, :a_len][::AUDIO_TOKEN_STRIDE][:AUDIO_TOKEN_MAX_LEN]
        torch.save(a_seq.float().cpu(), out_dir / "audio_tokens" / split / stem)

        t_len = int(text_attention_mask[i].sum().item())
        t_seq = t_tokens[i, :t_len]
        torch.save(t_seq.float().cpu(), out_dir / "text_tokens" / split / stem)


def process_batch_with_retry(batch: pd.DataFrame, split: str, *args, min_chunk: int = 1) -> None:
    """Runs process_batch, halving the batch on CUDA OOM (retrying after
    clearing the cache) down to `min_chunk` before giving up.
    """
    try:
        process_batch(batch, split, *args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) <= min_chunk:
            raise
        mid = len(batch) // 2
        print(f"[build_feature_cache] OOM on batch of {len(batch)}, retrying as two chunks of ~{mid}")
        process_batch_with_retry(batch.iloc[:mid], split, *args, min_chunk=min_chunk)
        process_batch_with_retry(batch.iloc[mid:], split, *args, min_chunk=min_chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--out-dir", default="data/meld/cache", type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build_feature_cache] device={device}")

    for subdir in CACHE_SUBDIRS:
        for split in args.splits:
            (args.out_dir / subdir / split).mkdir(parents=True, exist_ok=True)

    print("[build_feature_cache] loading backbones...")
    video_backbone = MViTv2Backbone().to(device)
    audio_backbone = Wav2Vec2Backbone().to(device)
    text_backbone = RobertaBackbone().to(device)
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    for split in args.splits:
        df = pd.read_parquet(args.processed_dir / f"{split}.parquet")
        remaining = df[
            ~df.apply(lambda r: _is_cached(args.out_dir, split, r.dialogue_id, r.utterance_id), axis=1)
        ].reset_index(drop=True)
        print(
            f"[build_feature_cache] split={split} utterances={len(df)} "
            f"already_cached={len(df) - len(remaining)} remaining={len(remaining)}"
        )

        # Sort by audio duration so within-batch padding (and hence conv1d
        # activation shapes) stays similar across consecutive batches. Highly
        # variable padded lengths batch-to-batch were causing CUDA allocator
        # fragmentation severe enough to exhaust the GPU over a full split.
        if len(remaining) > 0:
            durations = remaining["wav_path"].map(lambda p: sf.info(p).duration)
            remaining = remaining.iloc[durations.argsort().to_numpy()].reset_index(drop=True)

        start = time.time()
        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining.iloc[batch_start : batch_start + args.batch_size]
            process_batch_with_retry(
                batch, split, video_backbone, audio_backbone, text_backbone,
                audio_feature_extractor, tokenizer, device, args.out_dir,
            )
            torch.cuda.empty_cache()
            done = batch_start + len(batch)
            if done % (args.batch_size * 20) < args.batch_size or done == len(remaining):
                elapsed = time.time() - start
                print(f"[build_feature_cache] {split}: {done}/{len(remaining)} ({elapsed:.1f}s elapsed)")

        print(f"[build_feature_cache] split={split} done in {time.time() - start:.1f}s")

    print("[build_feature_cache] all splits complete.")


if __name__ == "__main__":
    main()
