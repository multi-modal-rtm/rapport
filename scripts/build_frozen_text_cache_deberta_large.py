"""Second-encoder replication, Step 2 prep (docs/PREREG_DeBERTa-v3-large.md):
runs the frozen `microsoft/deberta-v3-large` backbone (`DebertaV2Backbone`,
masked mean over the second-to-last hidden layer -- same convention as the
existing frozen RoBERTa "text" cache, cache_version 4) once over every
preprocessed MELD utterance (isolated, no context window -- mirrors the
frozen, non-contextual "text" cache scripts/build_feature_cache.py builds
for RoBERTa) and caches the pooled 1024-d vectors to
data/meld/cache/text_deberta_large/{split}/dia{d}_utt{u}.pt.

Video/audio caches are untouched and reused as-is (only the text
representation varies across this study's frozen-vs-fine-tuned /
encoder-family comparisons).

Usage:
    uv run python -m scripts.build_frozen_text_cache_deberta_large
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from transformers import AutoTokenizer

from rapport.data.cache_constants import TEXT_TOKEN_MAX_LEN
from rapport.models.backbones import DebertaV2Backbone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "microsoft/deberta-v3-large"
OUT_SUBDIR = "text_deberta_large"
CACHE_VERSION = "text_deberta_large_v1"
SPLITS = ("train", "dev", "test")
BATCH_SIZE = 32


def _is_cached(out_dir: Path, split: str, dialogue_id: int, utterance_id: int) -> bool:
    return (out_dir / split / f"dia{dialogue_id}_utt{utterance_id}.pt").exists()


@torch.no_grad()
def process_split(df: pd.DataFrame, split: str, backbone: DebertaV2Backbone, tokenizer, device: torch.device, out_dir: Path) -> int:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    remaining = [row for row in df.itertuples(index=False) if not _is_cached(out_dir, split, row.dialogue_id, row.utterance_id)]
    n_written = 0
    start = time.time()
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        text_inputs = tokenizer(
            [row.text for row in batch], padding=True, truncation=True, max_length=TEXT_TOKEN_MAX_LEN, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        pooled, _ = backbone(input_ids, attention_mask)

        for i, row in enumerate(batch):
            stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
            torch.save(pooled[i].float().cpu(), split_dir / stem)
            n_written += 1

        if (batch_start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - start
            print(f"[build_frozen_text_cache_deberta_large] {split}: {batch_start + len(batch)}/{len(remaining)} ({elapsed:.1f}s elapsed)", flush=True)

    return n_written


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build_frozen_text_cache_deberta_large] device={device} model={MODEL_NAME}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = DebertaV2Backbone(MODEL_NAME).to(device)
    backbone.eval()

    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    out_dir = PROJECT_ROOT / "data" / "meld" / "cache" / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLITS:
        df = pd.read_parquet(processed_dir / f"{split}.parquet")
        start = time.time()
        n = process_split(df, split, backbone, tokenizer, device, out_dir)
        counts[split] = n
        print(f"[build_frozen_text_cache_deberta_large] split={split} wrote {n} files in {time.time() - start:.1f}s", flush=True)

    manifest = {
        "cache_version": CACHE_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "pooling": "masked_mean_second_to_last_layer",
        "feature_dim": DebertaV2Backbone.OUTPUT_DIM,
        "context": "isolated utterance (no context window)",
        "splits": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_frozen_text_cache_deberta_large] wrote {out_dir / 'manifest.json'}: {manifest}", flush=True)


if __name__ == "__main__":
    main()
