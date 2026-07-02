# Project Pivot: SocialArcNet → RAPPORT

**Date:** 2026-07-02

## Background

The original plan was to extend our prior published model (SocialArcNet,
ICFNDS '25) and reproduce its 0.62 weighted F1 on MELD as "config A" before
building v2 on top of it. That reproduction effort (previous phase) landed at
0.3765 weighted F1 — see `docs/DISCREPANCY_REPORT_A_seed42.md`.

**The reproduction target is now cancelled.** The original SocialArcNet
training code was destroyed in a server wipe. The public GitHub repo contains
a wrong, older version of the code and **must be treated as untrusted
reference material** — not diffed against, not imported from.

We are instead building a new architecture, **RAPPORT**, whose thesis is that
**relationship dynamics (speaker-pair states)**, not individual speaker
states, are the primary representation for conversational emotion
recognition. SocialArcNet is now a cited literature baseline (published
number: 0.62 weighted F1) — not something we re-implement or reproduce.

## What changes

### 1. Gates

The `[0.60, 0.64]` reproduction gate is **void**. From the next phase
onward, validation is governed by two new criteria:

- **HARD FLOOR (stop gate):** any full multimodal model must beat a
  text-only logistic regression probe trained on cached RoBERTa features by
  **≥ 0.03 weighted F1**. This is a sanity floor, not an aspiration — if a
  multimodal model can't beat a trivial text-only linear probe by a
  meaningful margin, something is broken.
- **COMPETITIVENESS TARGET (not a stop gate):** ≥ 0.58 weighted F1 on MELD
  test for the model we intend to publish. Aspiration is 0.62+ (matching or
  beating the SocialArcNet baseline), but falling short of 0.58 does not by
  itself halt work — it's a target to steer toward, not a hard failure
  condition.

### 2. Renaming

The Python package is renamed `socialarcnet` → `rapport`
(`src/socialarcnet` → `src/rapport`), with `pyproject.toml`, all imports,
tests, and Hydra config references updated to match, in one atomic commit.
The project directory itself (`~/socialarcnet-v2`) is unchanged — only the
importable package name changes. The entrypoint becomes `python -m rapport`.

The v1 architecture (per-speaker GNN, mean-pooling, frozen backbones) is
**retained** as code but demoted to an internal ablation variant named
`speaker_only`. `configs/experiment/A_baseline.yaml` is renamed to
`speaker_only.yaml` (same architecture, new name). The old two-axis
(`temporal_attention` / `lora`) config scheme (`A_baseline`, `B_temporal`,
`C_lora`, `D_full`) is replaced by RAPPORT's three-component ablation scheme.
Experiment configs now expose four boolean flags: `relational`, `shift`,
`temporal`, `lora`.

New experiment config stubs (declarative flags only — model implementations
land in a later phase):

| config | relational | shift | temporal | lora | meaning |
|---|---|---|---|---|---|
| `speaker_only` | ✗ | ✗ | ✗ | ✗ | v1 architecture, frozen backbones (was `A_baseline`) |
| `speaker_only_lora` | ✗ | ✗ | ✗ | ✓ | v1 architecture, end-to-end LoRA |
| `full` | ✓ | ✓ | ✓ | ✗ | RAPPORT complete, frozen backbones |
| `minus_relational` | ✗ | ✓ | ✓ | ✗ | leave-one-out ablation |
| `minus_shift` | ✓ | ✗ | ✓ | ✗ | leave-one-out ablation |
| `minus_temporal` | ✓ | ✓ | ✗ | ✗ | leave-one-out ablation |
| `full_lora` | ✓ | ✓ | ✓ | ✓ | RAPPORT complete, end-to-end LoRA |

`run_name` interpolation (`${experiment.name}_seed${seed}`) is unchanged
structurally — it already derives from each config's `name` field, so the
new config names flow through automatically.

`src/rapport/__main__.py`'s "not implemented yet" guard now checks all four
flags (`relational`, `shift`, `temporal`, `lora`); only the all-flags-off
`speaker_only` architecture is implemented so far. Every other config is a
declarative stub until the corresponding model code is built.

### 3. Reference code

`reference/v1_UNTRUSTED/` is the designated location for the public
SocialArcNet v1 GitHub clone, if/when one is pulled onto this machine. It
does not exist on disk yet, so no rename was needed this phase. Policy for
whenever it does appear: keep it on disk for occasional human inspection,
but never diff RAPPORT code against it and never import from it — it is a
different (older, wrong) version of the model than what was actually
published and run.

### 4. Nothing already built was deleted

The MELD data pipeline (`scripts/download_meld.sh`,
`scripts/preprocess_meld.py`), the frozen-backbone feature cache
(`scripts/build_feature_cache.py`, `data/meld/cache/`), seed/repro infra
(`rapport.seed`, `rapport.repro`), the trainer (`rapport.training.trainer`),
and the broken `config A` / seed 42 run artifacts
(`outputs/A_baseline_seed42/`, `docs/DISCREPANCY_REPORT_A_seed42.md`) all
stay on disk untouched. The run artifacts are evidence for the upcoming
debugging phase, not a mistake to clean up.

### 5. Vision backbone

Since v1 comparability is no longer required, we're freed from
torchvision's `mvit_v2_s` fused-QKV constraint (its `MultiscaleAttention`
uses a single fused `qkv` linear layer, which complicates applying LoRA to
Q/V projections separately — the standard LoRA recipe assumes separate
projection matrices). **No backbone swap is happening yet.** This is noted
here as now-permitted; the actual decision (e.g. a move to an HF video
backbone with separate q/k/v projections, such as VideoMAE) is deferred to
the LoRA implementation phase, when it will actually matter.

## Status

Rename and config restructuring only. No training, no debugging this phase.
