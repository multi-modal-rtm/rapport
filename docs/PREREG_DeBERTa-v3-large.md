# Pre-registration: Second-Encoder Replication (DeBERTa-v3-large)

**Committed:** 2026-08-19, before any training for this experiment was run.

Reviewer-response experiment (Major Concern: narrow empirical base / RoBERTa-base
capacity). This is a **confirmatory** test of whether the paper's central
fine-tuning-boundary claim — that a residual graph stack adds null-to-negative
value once the text encoder itself is fine-tuned, while the same stack adds a
reliably positive value over a frozen text encoder — is an artifact of
RoBERTa-base's specific capacity, or holds under a substantially larger,
architecturally different encoder.

## Encoder choice (resolved before writing this file)

The task naming this replication conflicted internally: prose instructions
specified "roberta-large" throughout, while the committed file names specify
`DeBERTa-v3-large`. These are not interchangeable — RoBERTa-large's attention
modules are literally named `query`/`value` (the existing LoRA config applies
unmodified), while DeBERTa-v3's disentangled attention uses `query_proj`/
`key_proj`/`value_proj`. Asked directly, the requester chose **DeBERTa-v3-large**
(`microsoft/deberta-v3-large`, 24 layers, hidden size 1024, ~435M params) —
a different architecture family from RoBERTa entirely (disentangled
attention + relative position encoding, SentencePiece tokenizer), which makes
this a stronger test of encoder-family generality than a same-family
RoBERTa-large run would have been.

**Documented deviation from "same LoRA target modules (query, value)":**
DeBERTa-v3-large has no modules literally named `query`/`value`; the direct
analog — same role (query/value projections only, not key), same r=8/alpha=16/
dropout=0.05 — is `target_modules=["query_proj", "value_proj"]`. This was
verified to attach correctly (`peft.get_peft_model` finds and wraps both
per layer, 24 layers) and to keep the trainable fraction (LoRA adapters + head
only) at ~0.18% of total params, well inside the existing 5% adapter-tuning
guard (`MAX_TRAINABLE_FRACTION` in `src/rapport/models/text_classifier.py`).

**Documented correction to pooling layer:** the task text says "masked-mean
pooling of the second-to-last layer" for the fine-tuned foundation. Checked
against `docs/RECIPE.md`'s actual locked Phase T recipe and
`src/rapport/models/text_classifier.py`'s `ContextTextClassifier.encode`, the
**fine-tuned** foundation's locked recipe pools masked-mean over
`last_hidden_state` (the *final* layer) — second-to-last-layer pooling is the
separate, **frozen**-backbone convention (`RobertaBackbone` in
`src/rapport/models/backbones.py`, chosen for frozen-feature extraction
specifically because the final layer is over-specialized to the MLM
objective, `docs/DIAGNOSIS.md`). This replication uses each regime's actual
locked convention: last-hidden-state masked-mean for the fine-tuned DeBERTa
foundation (Steps 1/3 below), second-to-last-layer masked-mean for the frozen
DeBERTa feature anchor (Step 2), exactly mirroring which pooling rule governs
which regime in the base-model (RoBERTa) experiments already in this repo.

**New dependencies added** (`sentencepiece`, `protobuf`, via `uv add`,
committed alongside this file): required for DeBERTa-v3's SentencePiece
tokenizer, which RoBERTa's BPE tokenizer never needed. No existing dependency
version was changed.

## Hypothesis (pre-registered verbatim)

> The fine-tuning boundary replicates under a larger encoder. On a
> DeBERTa-v3-large fine-tuned foundation (MELD), the graph stack's paired
> residual contribution will be null-to-negative (not positive beyond seed
> noise), and the frozen-vs-fine-tuned difference-in-differences will remain
> positive with a 95% bootstrap CI excluding zero.

## Reading rule (pre-registered verbatim)

> If the paired gain is positive and its interval excludes zero, the boundary
> does NOT replicate and the paper's scope claim must narrow. Otherwise it
> replicates and strengthens the generality claim.

## Planned method (for audit against what actually ran)

1. **Foundation** (fine-tuned): `microsoft/deberta-v3-large` + LoRA
   (r=8, alpha=16, dropout=0.05, target=`[query_proj, value_proj]`),
   masked-mean pooling of `last_hidden_state`, plain CE, context k=8,
   max_length=256, lr=2e-4/linear-warmup-10%/max 10 epochs/patience 3 on val
   macro F1, bf16, batch by utterance (32) — every other Phase T hyperparameter
   from `docs/RECIPE.md` unchanged. Seeds {42, 1337, 2024}. New script (does
   not modify `scripts/train_context_text.py`, which stays the frozen
   RoBERTa-base Phase T script).
2. **Frozen control**: frozen (no gradient) DeBERTa-v3-large features,
   masked-mean pooled over the second-to-last hidden layer, feeding a linear
   probe trained with the RECIPE.md-locked GNN optimizer settings (mirrors
   `scripts/train_frozen_text_foundation.py`, single seed 42, matching that
   script's own convention of a one-time frozen-anchor build) — then the
   residual graph stack (`RapportModel(residual=True)`) on those frozen
   features at 3 graph-stack seeds {42, 1337, 2024}, mirroring
   `scripts/run_bridging_matrix.py`.
3. **Residual stack** on the fine-tuned foundation: `RapportModel` gains an
   additive `text_feature_dim` constructor parameter (default 768, so every
   existing RoBERTa-base config is byte-identical) so its fusion layer's
   input width and zero-init column slicing generalize to 1024-d text
   features; audio/video stay 768-d and their fusion columns stay zero-init.
   Full stack (relational=shift=temporal=residual=True), 3 graph-stack seeds,
   reading the DeBERTa-large text_ctx + z_text-logits cache built once from
   the seed-42 fine-tuned foundation (mirrors the existing convention that
   Phase N4's `text_ctx` cache is built once, not per-seed).
4. **Equality-at-init**: before launching any residual-stack training run,
   verify that a freshly constructed `RapportModel(residual=True,
   text_feature_dim=1024)` reproduces the cached DeBERTa z_text logits
   exactly (bit-for-bit `torch.equal`) on real cached data, for both
   `base_fusion_R`-equivalent and `full_R`-equivalent configs — the same
   property `tests/test_rapport_model_residual.py` already enforces for the
   768-d RoBERTa case.
5. **Difference-in-differences**: same independent-groups percentile
   bootstrap method as `docs/DIFF_IN_DIFF.md` (100,000 resamples, fixed RNG
   seed 20260818), applied to the DeBERTa-large frozen-regime paired diffs
   (n=3) vs. fine-tuned-regime paired diffs (n=3, k=8) — point estimate + 95%
   CI, and whether it excludes zero.

No existing value, cache, checkpoint, or result from any prior experiment is
read for anything other than methodological mirroring, and none is modified.

## Gate

- This file committed and pushed before any DeBERTa-v3-large training run.
- Equality-at-init passes on the large foundation before the residual-stack
  training run is launched.
- The verdict (replicates / does not replicate) is reported plainly against
  the reading rule above, whichever way it falls, in
  `docs/DeBERTa-v3-large_REPLICATION.md`.
