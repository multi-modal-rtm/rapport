"""Video backbone scouting experiment (diagnostics only — Fix-3-vs-swap decision input).

On a stratified ~1000-utterance train subsample + the full test split,
compares three video feature sources with the same linear probe:
  A. MViTv2-as-is       — the current cache (8 frames, square-resize transform, repeat-interleaved to 16)
  B. MViTv2-corrected   — same 8->16 frame duplication, but with the OFFICIAL
                          torchvision transform (resize shorter side, aspect-ratio-preserving, then center crop)
  C. VideoMAE-base      — native 16 frames (no duplication), Kinetics-finetuned checkpoint,
                          official VideoMAEImageProcessor preprocessing

Also audits whether the current preprocessing pipeline's hand-rolled resize
matches torchvision's official MViT_V2_S_Weights.transforms() and reports
any divergence (this is what motivates row B).

Frame decode/transform (CPU-bound) is parallelized across processes, same
pattern as scripts/preprocess_meld.py; backbone forward passes (GPU-bound)
run afterward in batches on the main process.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd
import psutil
import torch
import torchvision.transforms.v2.functional as TF
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torchvision.models.video import MViT_V2_S_Weights
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from rapport.data.constants import FRAME_MEAN, FRAME_SIZE, FRAME_STD, NUM_FRAMES
from rapport.models.backbones import MViTv2Backbone

RAW_DIR = Path("data/meld/raw/MELD.Raw")
VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
VIDEOMAE_NUM_FRAMES = 16
BATCH_SIZE = 32

# Native-resolution uint8 frames (row C, ~1280x720x3x16 bytes/clip = ~44MB) are
# what OOM'd the prior run: the old code decoded an entire split into an
# in-memory dict before any forward pass consumed it (~160GB for ~3600 clips).
# Streaming in small chunks (decode -> forward -> free) keeps the resident
# in-memory frame buffer bounded to CHUNK_SIZE clips regardless of split size.
CHUNK_SIZE = 64
_PROCESS = psutil.Process(os.getpid())


def log_rss(tag: str, n_clips_done: int) -> None:
    rss_gb = _PROCESS.memory_info().rss / 1e9
    print(f"[video_scouting][rss] {tag}: after {n_clips_done} clips, RSS={rss_gb:.2f} GB", flush=True)


def audit_transform_divergence() -> None:
    official = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    print("=== transform audit ===")
    print(f"official: resize_size={official.resize_size} crop_size={official.crop_size} "
          f"mean={official.mean} std={official.std}")
    print("current preprocess_meld.py: TF.resize(frames, size=[256, 256], ...) then center_crop([224,224])")
    print(f"current mean={FRAME_MEAN} std={FRAME_STD}")
    print(
        "DIVERGENCE FOUND: official resize_size=[256] (single value) resizes the SHORTER edge to 256, "
        "preserving aspect ratio, before center-cropping to 224x224. The current pipeline calls "
        "TF.resize(..., size=[256, 256]) -- a two-element size, which torchvision resizes to EXACTLY "
        "256x256, ignoring aspect ratio. MELD source clips are 1280x720 (16:9), so this distorts "
        "(vertically stretches) every frame before the model ever sees it. mean/std match exactly."
    )


def find_split_video_dir(split: str) -> Path:
    for candidate in sorted(RAW_DIR.glob(f"**/*{split}*")):
        if candidate.is_dir() and next(candidate.glob("dia*_utt*.mp4"), None) is not None:
            return candidate
    raise FileNotFoundError(f"no video dir for split={split}")


def _read_native_frames(mp4_path: str, num_frames: int) -> np.ndarray:
    """Uniformly samples num_frames frames -> [num_frames, H, W, C] uint8."""
    container = av.open(mp4_path)
    try:
        stream = container.streams.video[0]
        all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    finally:
        container.close()
    total = len(all_frames)
    idx = np.linspace(0, total - 1, num=num_frames).round().astype(int).clip(0, total - 1)
    return np.stack([all_frames[i] for i in idx])


def _worker_row_b(args: tuple[str, int, int]) -> tuple[int, int, np.ndarray]:
    """CPU-only worker: decode + apply the OFFICIAL (aspect-preserving) transform. Returns normalized [8,3,224,224]."""
    mp4_path, dialogue_id, utterance_id = args
    native = torch.from_numpy(_read_native_frames(mp4_path, NUM_FRAMES))
    frames = native.permute(0, 3, 1, 2).float() / 255.0
    frames = TF.resize(frames, size=[256], antialias=True)  # single value: shorter edge -> 256, aspect preserved
    frames = TF.center_crop(frames, output_size=[FRAME_SIZE, FRAME_SIZE])
    mean = torch.tensor(FRAME_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(FRAME_STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return dialogue_id, utterance_id, frames.numpy()


def _worker_row_c(args: tuple[str, int, int]) -> tuple[int, int, np.ndarray]:
    """CPU-only worker: decode 16 native frames (uint8, HWC) for VideoMAEImageProcessor."""
    mp4_path, dialogue_id, utterance_id = args
    native = _read_native_frames(mp4_path, VIDEOMAE_NUM_FRAMES)
    return dialogue_id, utterance_id, native


def _decode_chunk(chunk_rows: list, video_dir: Path, worker_fn, pool: ProcessPoolExecutor) -> dict[tuple[int, int], np.ndarray]:
    tasks = [
        (str(video_dir / f"dia{row.dialogue_id}_utt{row.utterance_id}.mp4"), row.dialogue_id, row.utterance_id)
        for row in chunk_rows
    ]
    results: dict[tuple[int, int], np.ndarray] = {}
    futures = [pool.submit(worker_fn, t) for t in tasks]
    for fut in as_completed(futures):
        dia, utt, arr = fut.result()
        results[(dia, utt)] = arr
    return results


def stream_extract(
    df: pd.DataFrame, video_dir: Path, decode_worker_fn, forward_fn, decode_pool: ProcessPoolExecutor,
    chunk_size: int = CHUNK_SIZE, log_tag: str = "",
) -> np.ndarray:
    """Streams decode -> forward -> free over df in chunks of chunk_size clips,
    so the resident decoded-frame buffer never holds more than one chunk's
    worth of clips regardless of split size (this is what the prior OOM'd
    version got wrong: it decoded the entire split into memory up front).

    decode_pool must be a spawn-context ProcessPoolExecutor created before any
    CUDA context exists in this process (see main()) — forking a process that
    already holds a CUDA context or initialized OpenMP threads segfaults the
    child workers (observed directly: synchronized libgomp segfaults in
    journalctl the instant a fork-context pool was spawned after
    `MViTv2Backbone().to(device)` had already run).
    """
    rows = list(df.itertuples(index=False))
    feats = []
    n_done = 0
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start : start + chunk_size]
        decoded = _decode_chunk(chunk_rows, video_dir, decode_worker_fn, decode_pool)
        pooled = forward_fn(chunk_rows, decoded)
        feats.append(pooled)

        prev_done = n_done
        n_done += len(chunk_rows)
        del decoded, pooled, chunk_rows
        gc.collect()
        torch.cuda.empty_cache()

        if n_done // 100 > prev_done // 100 or n_done == len(rows):
            log_rss(f"{log_tag} chunk", n_done)

    return np.concatenate(feats)


def load_videomae_with_bias_fix(model_name: str) -> VideoMAEForVideoClassification:
    """VideoMAE Hub checkpoints use fused q_bias/v_bias naming that this
    transformers version doesn't auto-map, silently zero-initializing all
    attention biases. Remaps them from the raw checkpoint. See docs/DIAGNOSIS.md.
    """
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    ckpt_path = hf_hub_download(model_name, "pytorch_model.bin")
    raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    fixed_state = model.state_dict()
    n_remapped = 0
    for key, value in raw_sd.items():
        if key.endswith(".attention.attention.q_bias"):
            fixed_state[key.replace(".q_bias", ".query.bias")] = value
            n_remapped += 1
        elif key.endswith(".attention.attention.v_bias"):
            fixed_state[key.replace(".v_bias", ".value.bias")] = value
            n_remapped += 1
    model.load_state_dict(fixed_state)
    print(f"[video_scouting] VideoMAE bias fix: remapped {n_remapped} params")
    return model


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    parts = []
    for _, group in df.groupby("label"):
        k = max(1, round(len(group) * frac))
        idx = rng.choice(len(group), size=min(k, len(group)), replace=False)
        parts.append(group.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def probe_unbalanced(X_train, y_train, X_test, y_test, max_iter: int = 2000) -> float:
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    return f1_score(y_test, clf.predict(X_test), average="weighted", zero_division=0)


@torch.no_grad()
def extract_row_a_cached(df: pd.DataFrame, cache_dir: Path, split: str) -> np.ndarray:
    feats = []
    for row in df.itertuples(index=False):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        feats.append(torch.load(cache_dir / "video" / split / stem, weights_only=True).numpy())
    return np.stack(feats)


@torch.no_grad()
def mvit_forward_chunk(chunk_rows: list, decoded: dict, backbone: MViTv2Backbone, device) -> np.ndarray:
    feats = []
    keys = [(row.dialogue_id, row.utterance_id) for row in chunk_rows]
    for start in range(0, len(keys), BATCH_SIZE):
        sub = keys[start : start + BATCH_SIZE]
        frames = torch.stack([torch.from_numpy(decoded[k]) for k in sub]).to(device)
        pooled, _ = backbone(frames)
        feats.append(pooled.cpu().numpy())
        del frames, pooled
    return np.concatenate(feats)


@torch.no_grad()
def videomae_forward_chunk(
    chunk_rows: list, decoded: dict, model: VideoMAEForVideoClassification,
    processor: VideoMAEImageProcessor, device,
) -> np.ndarray:
    encoder = model.videomae.to(device)
    feats = []
    keys = [(row.dialogue_id, row.utterance_id) for row in chunk_rows]
    for start in range(0, len(keys), BATCH_SIZE):
        sub = keys[start : start + BATCH_SIZE]
        batch_videos = [[frame for frame in decoded[k]] for k in sub]
        inputs = processor(batch_videos, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = encoder(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        feats.append(pooled.cpu().numpy())
        del batch_videos, inputs, pixel_values, outputs, pooled
    return np.concatenate(feats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/meld/processed", type=Path)
    parser.add_argument("--cache-dir", default="data/meld/cache", type=Path)
    parser.add_argument("--train-subsample", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--chunk-size", default=CHUNK_SIZE, type=int)
    args = parser.parse_args()

    audit_transform_divergence()
    log_rss("startup", 0)

    # Spawn-context pool created before any CUDA context (or heavily-threaded
    # BLAS/OpenMP state) exists in this process. A default fork-context pool
    # created *after* `MViTv2Backbone().to(device)` reproducibly segfaulted
    # every worker (synchronized libgomp segfault, confirmed via journalctl)
    # the instant it was spawned — forking a CUDA-initialized process is
    # unsafe. One pool is reused across every chunk/row below so we pay the
    # spawn cost (fresh interpreter per worker) only once, not per chunk.
    decode_pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_df_full = pd.read_parquet(args.processed_dir / "train.parquet")
        train_df = stratified_subsample(train_df_full, args.train_subsample, args.seed)
        test_df = pd.read_parquet(args.processed_dir / "test.parquet")
        print(f"\n[video_scouting] train subsample: {len(train_df)} (stratified from {len(train_df_full)}), "
              f"test: {len(test_df)} (full)")

        train_video_dir = find_split_video_dir("train")
        test_video_dir = find_split_video_dir("test")

        y_train = train_df["label"].to_numpy()
        y_test = test_df["label"].to_numpy()

        results = {}

        print("\n[video_scouting] === Row A: MViTv2-as-is (from cache) ===")
        A_train = extract_row_a_cached(train_df, args.cache_dir, "train")
        A_test = extract_row_a_cached(test_df, args.cache_dir, "test")
        results["A_mvitv2_as_is"] = probe_unbalanced(A_train, y_train, A_test, y_test)
        print(f"[video_scouting] Row A unbalanced weighted F1 = {results['A_mvitv2_as_is']:.4f}")
        del A_train, A_test
        gc.collect()
        log_rss("after row A", len(train_df) + len(test_df))

        print("\n[video_scouting] === Row B: MViTv2 with corrected transform (streaming) ===")
        mvit_backbone = MViTv2Backbone().to(device)
        B_train = stream_extract(
            train_df, train_video_dir, _worker_row_b,
            lambda rows, dec: mvit_forward_chunk(rows, dec, mvit_backbone, device),
            decode_pool, args.chunk_size, log_tag="row B train",
        )
        B_test = stream_extract(
            test_df, test_video_dir, _worker_row_b,
            lambda rows, dec: mvit_forward_chunk(rows, dec, mvit_backbone, device),
            decode_pool, args.chunk_size, log_tag="row B test",
        )
        results["B_mvitv2_corrected"] = probe_unbalanced(B_train, y_train, B_test, y_test)
        print(f"[video_scouting] Row B unbalanced weighted F1 = {results['B_mvitv2_corrected']:.4f}")
        del mvit_backbone, B_train, B_test
        gc.collect()
        torch.cuda.empty_cache()
        log_rss("after row B", len(train_df) + len(test_df))

        print("\n[video_scouting] === Row C: VideoMAE-base (native 16 frames, streaming) ===")
        videomae_model = load_videomae_with_bias_fix(VIDEOMAE_MODEL_NAME)
        videomae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL_NAME)
        C_train = stream_extract(
            train_df, train_video_dir, _worker_row_c,
            lambda rows, dec: videomae_forward_chunk(rows, dec, videomae_model, videomae_processor, device),
            decode_pool, args.chunk_size, log_tag="row C train",
        )
        C_test = stream_extract(
            test_df, test_video_dir, _worker_row_c,
            lambda rows, dec: videomae_forward_chunk(rows, dec, videomae_model, videomae_processor, device),
            decode_pool, args.chunk_size, log_tag="row C test",
        )
        results["C_videomae"] = probe_unbalanced(C_train, y_train, C_test, y_test)
        print(f"[video_scouting] Row C unbalanced weighted F1 = {results['C_videomae']:.4f}")
        del videomae_model, videomae_processor, C_train, C_test
        gc.collect()
        torch.cuda.empty_cache()
        log_rss("after row C", len(train_df) + len(test_df))

        print("\n[video_scouting] === summary (unbalanced weighted F1, same subsample) ===")
        for name, wf1 in results.items():
            print(f"  {name}: {wf1:.4f}")
    finally:
        decode_pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
