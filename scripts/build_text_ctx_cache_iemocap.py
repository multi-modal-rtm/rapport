"""Phase N5-B Step B5.0: caches the frozen IEMOCAP text anchor's pooled
768-d contextual embedding + 7-d(6-d) classifier logits for every IEMOCAP
utterance, all splits, to data/iemocap/cache/text_ctx_iemocap/{split}/... +
data/iemocap/cache/text_ctx_iemocap_logits/{split}/... -- the IEMOCAP
analog of scripts/build_text_ctx_cache.py (that script itself untouched;
this is a parallel copy repointed at data/iemocap/processed and the
6-class label set, same pattern as train_context_text_iemocap.py).

The encoder is FROZEN at the IEMOCAP text anchor's seed-42 checkpoint
(outputs/iemocap_text_anchor_seed42/best_model.pt, docs/PHASE_N5B.md Step
B4) for the whole of this phase -- one-time build, not per-seed.

Usage:
    uv run python -m scripts.build_text_ctx_cache_iemocap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rapport.data.constants import IEMOCAP_EMOTION_LABELS
from rapport.data.context_text import ContextTextCollator, MELDContextTextDataset, DEFAULT_K, DEFAULT_MAX_LENGTH
from rapport.models.text_classifier import ContextTextClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = len(IEMOCAP_EMOTION_LABELS)
PROCESSED_DIR = PROJECT_ROOT / "data" / "iemocap" / "processed"

FROZEN_CHECKPOINT = PROJECT_ROOT / "outputs" / "iemocap_text_anchor_seed42" / "best_model.pt"
CACHE_VERSION = "text_ctx_iemocap_v1"
SPLIT_FILENAMES = ("train", "val", "test")  # loader-internal name -> ("dev" slot maps to "val" file)
BATCH_SIZE = 64


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_encoder(device: torch.device, checkpoint_path: Path = FROZEN_CHECKPOINT) -> ContextTextClassifier:
    model = ContextTextClassifier(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def build_split_cache(
    model: ContextTextClassifier,
    tokenizer,
    split: str,
    out_dir: Path,
    logits_out_dir: Path,
    device: torch.device,
    k: int = DEFAULT_K,
) -> int:
    dataset = MELDContextTextDataset(PROCESSED_DIR / f"{split}.parquet", tokenizer, k=k, max_length=DEFAULT_MAX_LENGTH)
    collate = ContextTextCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4)

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    logits_split_dir = logits_out_dir / split
    logits_split_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for batch_start, batch in zip(range(0, len(dataset), BATCH_SIZE), loader):
        batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        pooled = model.encode(batch_gpu["input_ids"], batch_gpu["attention_mask"])
        logits = model.classifier(pooled)

        for i in range(pooled.shape[0]):
            dialogue_id, t = dataset.index[batch_start + i]
            utterance_id = int(dataset.dialogues[dialogue_id]["utterance_id"].iloc[t])
            stem = f"dia{dialogue_id}_utt{utterance_id}.pt"
            torch.save(pooled[i].float().cpu(), split_dir / stem)
            torch.save(logits[i].float().cpu(), logits_split_dir / stem)
            n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out-subdir", type=str, default="text_ctx_iemocap")
    parser.add_argument("--checkpoint", type=Path, default=FROZEN_CHECKPOINT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build_text_ctx_cache_iemocap] device={device} checkpoint={args.checkpoint} k={args.k}")
    checkpoint_hash = sha256_of(args.checkpoint)
    print(f"[build_text_ctx_cache_iemocap] checkpoint sha256={checkpoint_hash}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = load_frozen_encoder(device, checkpoint_path=args.checkpoint)

    out_dir = PROJECT_ROOT / "data" / "iemocap" / "cache" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_out_dir = PROJECT_ROOT / "data" / "iemocap" / "cache" / f"{args.out_subdir}_logits"
    logits_out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLIT_FILENAMES:
        start = time.time()
        n = build_split_cache(model, tokenizer, split, out_dir, logits_out_dir, device, k=args.k)
        counts[split] = n
        print(f"[build_text_ctx_cache_iemocap] split={split} wrote {n} files in {time.time() - start:.1f}s")

    manifest = {
        "cache_version": CACHE_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(Path(args.checkpoint).resolve().relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "k": args.k,
        "max_length": DEFAULT_MAX_LENGTH,
        "pooling": "masked_mean_last_hidden_state",
        "logits_cache_subdir": f"{args.out_subdir}_logits",
        "splits": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_text_ctx_cache_iemocap] wrote {out_dir / 'manifest.json'}: {manifest}")


if __name__ == "__main__":
    main()
