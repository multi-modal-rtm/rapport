"""Phase N5-B Step B5.0 verification: process-independence check for the
IEMOCAP text_ctx_iemocap cache -- IEMOCAP analog of
scripts/verify_text_ctx_cache.py (untouched; this is a parallel copy).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from rapport.data.context_text import DEFAULT_K, DEFAULT_MAX_LENGTH, encode_with_context
from scripts.build_text_ctx_cache_iemocap import load_frozen_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "iemocap" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "iemocap" / "cache" / "text_ctx_iemocap"


def process_independence_check(n_samples: int = 20, seed: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = load_frozen_encoder(device)  # fresh process, fresh instance

    rng = random.Random(seed)
    max_diffs = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet").sort_values(["dialogue_id", "utterance_id"])
        dialogues = {did: g.reset_index(drop=True) for did, g in df.groupby("dialogue_id", sort=True)}
        per_split = n_samples // 3 + 1
        sampled = 0
        dialogue_ids = list(dialogues.keys())
        rng.shuffle(dialogue_ids)
        for did in dialogue_ids:
            if sampled >= per_split:
                break
            g = dialogues[did]
            t = rng.randrange(len(g))
            with torch.no_grad():
                encoded = encode_with_context(
                    tokenizer, g["speaker"].tolist(), g["text"].tolist(), t, DEFAULT_K, DEFAULT_MAX_LENGTH
                )
                live = model.encode(
                    encoded["input_ids"].unsqueeze(0).to(device),
                    encoded["attention_mask"].unsqueeze(0).to(device),
                )[0].float().cpu()

            uid = int(g["utterance_id"].iloc[t])
            cached = torch.load(CACHE_DIR / split / f"dia{did}_utt{uid}.pt", weights_only=True)
            diff = (live - cached).abs().max().item()
            max_diffs.append(diff)
            ok = torch.allclose(live, cached, atol=1e-4, rtol=1e-4)
            print(f"  split={split} dia={did} utt={uid}: max_abs_diff={diff:.6f} allclose={ok}")
            sampled += 1

    overall_max = max(max_diffs)
    print(f"[process_independence_check] {len(max_diffs)} samples, overall max_abs_diff={overall_max:.6f}")
    assert overall_max < 1e-3, "process-independence check FAILED -- cache does not match a fresh-process recompute"
    print("[process_independence_check] PASS")


if __name__ == "__main__":
    print("=== process-independence check (IEMOCAP text_ctx cache) ===")
    process_independence_check()
