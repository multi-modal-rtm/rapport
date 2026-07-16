"""Phase N4 Step 0: caches the frozen Phase T text encoder's pooled 768-d
contextual embedding (`ContextTextClassifier.encode`, masked mean of
last_hidden_state, k=8 context window per docs/PHASE_T.md) for every MELD
utterance, all splits, to data/meld/cache/text_ctx/{split}/dia{d}_utt{u}.pt.
Also caches the SAME model's 7-d classifier logits (z_text) alongside it,
to data/meld/cache/{out-subdir}_logits/{split}/... -- Phase N4-R's
residual redesign (spec v1.1, docs/PHASE_N4R.md) adds these to the
multimodal stack's own logits rather than starting from scratch.

The encoder is FROZEN at the Phase T seed-42 (plain-CE recipe) best
checkpoint (`outputs/context_text_plain_ce_seed42/best_model.pt`) for the
whole of Phase N4 -- this is a one-time build, not per-seed. Downstream
ablation-matrix seed variation comes from the new fusion/relational/shift
head's init and data order, not from re-encoding text.

Usage:
    uv run python scripts/build_text_ctx_cache.py
    uv run python scripts/build_text_ctx_cache.py --k 0 --out-subdir text_ctx_k0

`--k`/`--out-subdir` support Phase N4-R Step 2's redundancy probe: the SAME
frozen encoder, fed context-free (k=0) input instead of the default k=8,
written to a separate cache subdir so it never collides with the k=8 cache
every other Phase N4 config depends on.
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

from rapport.data.constants import EMOTION_LABELS
from rapport.data.context_text import ContextTextCollator, MELDContextTextDataset, DEFAULT_K, DEFAULT_MAX_LENGTH
from rapport.models.text_classifier import ContextTextClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = len(EMOTION_LABELS)

FROZEN_CHECKPOINT = PROJECT_ROOT / "outputs" / "context_text_plain_ce_seed42" / "best_model.pt"
CACHE_VERSION = "text_ctx_v1"
SPLITS = ("train", "dev", "test")
BATCH_SIZE = 64


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_encoder(device: torch.device) -> ContextTextClassifier:
    model = ContextTextClassifier(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(FROZEN_CHECKPOINT, map_location=device, weights_only=True)
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
    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    dataset = MELDContextTextDataset(processed_dir / f"{split}.parquet", tokenizer, k=k, max_length=DEFAULT_MAX_LENGTH)
    collate = ContextTextCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4)

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    logits_split_dir = logits_out_dir / split
    logits_split_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for batch_start, batch in zip(range(0, len(dataset), BATCH_SIZE), loader):
        batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        pooled = model.encode(batch_gpu["input_ids"], batch_gpu["attention_mask"])  # [B, 768]
        logits = model.classifier(pooled)  # [B, num_classes] -- z_text, spec v1.1

        for i in range(pooled.shape[0]):
            dialogue_id, t = dataset.index[batch_start + i]
            # t is the POSITIONAL index within the dialogue, not the raw
            # utterance_id -- some dialogues have gaps in utterance_id (bad
            # clips excluded upstream; verified directly: 73/1038 train
            # dialogues have non-contiguous utterance_id sequences), so the
            # cache filename must use the real column value, not t.
            utterance_id = int(dataset.dialogues[dialogue_id]["utterance_id"].iloc[t])
            stem = f"dia{dialogue_id}_utt{utterance_id}.pt"
            torch.save(pooled[i].float().cpu(), split_dir / stem)
            torch.save(logits[i].float().cpu(), logits_split_dir / stem)
            n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out-subdir", type=str, default="text_ctx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build_text_ctx_cache] device={device} checkpoint={FROZEN_CHECKPOINT} k={args.k}")
    checkpoint_hash = sha256_of(FROZEN_CHECKPOINT)
    print(f"[build_text_ctx_cache] checkpoint sha256={checkpoint_hash}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = load_frozen_encoder(device)

    out_dir = PROJECT_ROOT / "data" / "meld" / "cache" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_out_dir = PROJECT_ROOT / "data" / "meld" / "cache" / f"{args.out_subdir}_logits"
    logits_out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in SPLITS:
        start = time.time()
        n = build_split_cache(model, tokenizer, split, out_dir, logits_out_dir, device, k=args.k)
        counts[split] = n
        print(f"[build_text_ctx_cache] split={split} wrote {n} files in {time.time() - start:.1f}s")

    manifest = {
        "cache_version": CACHE_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(FROZEN_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "k": args.k,
        "max_length": DEFAULT_MAX_LENGTH,
        "pooling": "masked_mean_last_hidden_state",
        "logits_cache_subdir": f"{args.out_subdir}_logits",
        "splits": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_text_ctx_cache] wrote {out_dir / 'manifest.json'}: {manifest}")


if __name__ == "__main__":
    main()
