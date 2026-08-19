"""Second-encoder replication, Steps 2b/3 (docs/PREREG_DeBERTa-v3-large.md):
runs the Full residual stack (relational=shift=temporal=residual=True) on
BOTH the frozen and the fine-tuned DeBERTa-v3-large text representations,
3 graph-stack seeds each -- mirrors scripts/run_bridging_matrix.py's
methodology, generalized via RapportModel's `text_feature_dim=1024`.

Scope note: the base-model bridging experiment also ran a `base_fusion_R`
(non-relational/shift/temporal) config; this replication runs only the
Full-stack condition (matching Step 3's "Full stack" requirement and the
paper's headline paired-gain comparison) to keep this reviewer-response
replication's scope to what the pre-registered hypothesis actually needs.

Before training anything, verifies equality-at-init on REAL cached data for
both the frozen and fine-tuned conditions (a fresh residual model's logits
must equal the cached z_text exactly) -- the gate condition in
docs/PREREG_DeBERTa-v3-large.md.

Usage:
    uv run python -m scripts.run_deberta_large_residual_matrix
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from rapport.models.rapport_model import RapportModel
from scripts.train_rapport import train as train_rapport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1337, 2024)
NUM_CLASSES = 7
TEXT_FEATURE_DIM = 1024

CONDITIONS = {
    # config_name_prefix: text_cache_subdir
    "full_R_deberta_large_frozen": "text_deberta_large",
    "full_R_deberta_large_finetuned": "text_ctx_deberta_large",
}

FULL_STACK_FLAGS = {"relational": True, "shift": True, "temporal": True, "residual": True}


def verify_equality_at_init(text_cache_subdir: str, condition_label: str) -> None:
    """Real-cache analog of tests/test_rapport_model_residual.py's
    `test_residual_equals_frozen_text_foundation_logits_on_real_cache`, run
    here (not just in the unit-test suite) as an explicit pre-launch gate
    check for this specific cache, per the pre-registration's gate
    condition."""
    processed_dir = PROJECT_ROOT / "data" / "meld" / "processed"
    cache_dir = PROJECT_ROOT / "data" / "meld" / "cache"
    df = pd.read_parquet(processed_dir / "test.parquet")
    dialogue_id = int(df["dialogue_id"].iloc[0])
    rows = df[df["dialogue_id"] == dialogue_id].sort_values("utterance_id")
    assert len(rows) > 0

    video_feat, audio_feat, text_feat, text_logits = [], [], [], []
    for row in rows.itertuples(index=False):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        video_feat.append(torch.load(cache_dir / "video" / "test" / stem, weights_only=True))
        audio_feat.append(torch.load(cache_dir / "audio" / "test" / stem, weights_only=True))
        text_feat.append(torch.load(cache_dir / text_cache_subdir / "test" / stem, weights_only=True))
        text_logits.append(torch.load(cache_dir / f"{text_cache_subdir}_logits" / "test" / stem, weights_only=True))

    video_feat = torch.stack(video_feat).unsqueeze(0)
    audio_feat = torch.stack(audio_feat).unsqueeze(0)
    text_feat = torch.stack(text_feat).unsqueeze(0)
    text_logits = torch.stack(text_logits).unsqueeze(0)
    length = len(rows)
    dialogue_mask = torch.ones(1, length, dtype=torch.bool)
    speaker_ids = torch.tensor(rows["speaker_id"].tolist(), dtype=torch.long).unsqueeze(0)

    # FULL_STACK_FLAGS has temporal=True -- load real video/audio token
    # sequences too (mirrors tests/test_rapport_model_residual.py's
    # real-cache equality test).
    video_tokens, audio_tokens = [], []
    for row in rows.itertuples(index=False):
        stem = f"dia{row.dialogue_id}_utt{row.utterance_id}.pt"
        video_tokens.append(torch.load(cache_dir / "video_tokens" / "test" / stem, weights_only=True))
        audio_tokens.append(torch.load(cache_dir / "audio_tokens" / "test" / stem, weights_only=True))
    video_token_len = video_tokens[0].shape[0]
    audio_max_len = max(a.shape[0] for a in audio_tokens)
    audio_padded = torch.stack(
        [torch.nn.functional.pad(a, (0, 0, 0, audio_max_len - a.shape[0])) for a in audio_tokens]
    ).unsqueeze(0)
    audio_mask = torch.zeros(1, length, audio_max_len, dtype=torch.bool)
    for i, a in enumerate(audio_tokens):
        audio_mask[0, i, : a.shape[0]] = True

    torch.manual_seed(0)
    model = RapportModel(
        num_classes=NUM_CLASSES, text_feature_dim=TEXT_FEATURE_DIM, **FULL_STACK_FLAGS
    )
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            video_feat, audio_feat, text_feat, speaker_ids, dialogue_mask,
            video_tokens=torch.stack(video_tokens).unsqueeze(0),
            video_tokens_mask=torch.ones(1, length, video_token_len, dtype=torch.bool),
            audio_tokens=audio_padded,
            audio_tokens_mask=audio_mask,
            text_logits=text_logits,
        )

    assert torch.equal(logits, text_logits), (
        f"[GATE FAILURE] {condition_label} (text_cache_subdir={text_cache_subdir}): fresh residual model's "
        "logits do not exactly equal cached z_text at init on real data"
    )
    print(f"[equality-at-init] PASS: {condition_label} (text_cache_subdir={text_cache_subdir}, dialogue_id={dialogue_id})", flush=True)


def run(run_name: str, text_cache_subdir: str, seed: int) -> dict:
    run_dir = PROJECT_ROOT / "outputs" / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {run_name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text())

    print(f"[run] {run_name} (text_cache_subdir={text_cache_subdir}, seed={seed}, text_feature_dim={TEXT_FEATURE_DIM})", flush=True)
    start = time.time()
    report = train_rapport(
        seed=seed, run_dir=run_dir, text_cache_subdir=text_cache_subdir,
        text_feature_dim=TEXT_FEATURE_DIM, **FULL_STACK_FLAGS,
    )
    print(f"[run-done] {run_name} wall_clock_sec={time.time() - start:.1f}", flush=True)
    return report


def main() -> None:
    for condition_prefix, text_cache_subdir in CONDITIONS.items():
        verify_equality_at_init(text_cache_subdir, condition_prefix)

    for condition_prefix, text_cache_subdir in CONDITIONS.items():
        for seed in SEEDS:
            run_name = f"{condition_prefix}_seed{seed}"
            report = run(run_name, text_cache_subdir, seed)
            print(
                f"[status] {run_name} test_weighted_f1={report['test_weighted_f1']:.4f} "
                f"all_7_nonzero={report['all_7_classes_nonzero']}",
                flush=True,
            )

    print("[done] DeBERTa-v3-large residual matrix complete", flush=True)


if __name__ == "__main__":
    main()
