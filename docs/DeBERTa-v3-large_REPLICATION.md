# Second-Encoder Replication: DeBERTa-v3-large

Reviewer-response experiment (Major Concern: narrow empirical base /
RoBERTa-base capacity), pre-registered in `docs/PREREG_DeBERTa-v3-large.md`
before any training ran. This confirmatory test asks whether the paper's
central claim — a residual graph stack adds reliably positive value over a
**frozen** text encoder but null-to-negative value over a **fine-tuned** one
— holds under a substantially larger, architecturally different encoder
(`microsoft/deberta-v3-large`, 24 layers, hidden 1024, disentangled
attention, ~435M params) rather than being an artifact of RoBERTa-base's
specific capacity (12 layers, hidden 768, ~125M params).

**No existing stored value, checkpoint, or cache from any prior experiment
was read for anything beyond methodological mirroring, and none was
modified.** All results below are new runs under new `outputs/*_deberta_*`
and `data/meld/cache/*_deberta_large*` paths.

## Verdict

**DOES NOT REPLICATE**, against the pre-registered reading rule, stated
plainly:

> If the paired gain is positive and its interval excludes zero, the
> boundary does NOT replicate and the paper's scope claim must narrow.

On the DeBERTa-v3-large fine-tuned foundation, the Full stack's paired gain
over the fine-tuned text anchor is **positive at all 3 seeds**
(+0.0044, +0.0012, +0.0060; mean **+0.0039**), and a percentile bootstrap
interval on those 3 per-seed diffs is **[+0.0012, +0.0060]**, which excludes
zero. Under RoBERTa-base, the equivalent fine-tuned-regime paired gain was
null-to-negative (`docs/NONZERO_INIT_CONTROL.md`: mean −0.0060, seeds
{+0.0021, −0.0168, −0.0031}; `docs/DIFF_IN_DIFF.md`'s k=8 n=7 sample: mean
≈ −0.0088). Under DeBERTa-v3-large, the sign flipped: every seed showed a
small but positive gain. By the letter of the pre-registered rule, this
means the fine-tuning boundary as originally stated does **not** hold
unconditionally across encoder families, and the paper's scope claim should
narrow (e.g. to "observed for RoBERTa-base-scale fine-tuned encoders" rather
than encoder-family-general) unless the caveat below changes how this is
weighed.

**Caveat, stated as plainly as the result above**: this fine-tuned-regime
gain, while consistently positive, is an order of magnitude smaller than the
frozen-regime gain (+0.0039 vs. +0.0227 mean, roughly 6x smaller) and is
based on **n=3 seeds** — the interval above is a percentile bootstrap over
only 3 points, so (as `docs/DIFF_IN_DIFF.md` already cautions for its
similarly small n=3 frozen-regime sample) the achievable bootstrap
percentiles are constrained to near the 3 raw values themselves; with only
3 points, "excludes zero" is close to "all 3 raw seeds share the same sign,"
a much weaker statement than the base paper's n=7 fine-tuned sample
supports. The direction is consistent and the effect is real by this test,
but its practical/statistical strength is not comparable to the n=7 base
result, and a reader should weigh it accordingly.

The **frozen-regime vs. fine-tuned-regime difference-in-differences**
component of the hypothesis, separately, **does** replicate: DiD = **+0.0188**,
95% CI **[+0.0144, +0.0239]**, excludes zero (100.00% of bootstrap resamples
have frozen gain exceeding fine-tuned gain) — the frozen-regime gain remains
reliably larger than the fine-tuned-regime gain under DeBERTa-v3-large, same
as under RoBERTa-base. What does not replicate is the stronger sub-claim
that the fine-tuned-regime gain itself is null-to-negative.

## Results

### Large text anchor (Step 1: fine-tuned DeBERTa-v3-large foundation)

| seed | test weighted F1 | test macro F1 | best epoch | epoch time (avg) |
|---|---|---|---|---|
| 42 | 0.6596 | 0.4733 | 5 | 168.8s |
| 1337 | 0.6622 | 0.5092 | 9 | 169.1s |
| 2024 | 0.6567 | 0.4806 | 8 | 168.7s |
| **mean** | **0.6595** | 0.4877 | | |

All 3 seeds have all 7 emotion classes nonzero on test. For comparison, the
RoBERTa-base fine-tuned anchor (`context_text_plain_ce_seed{42,1337,2024}`,
unchanged, read only for comparison) is in the ~0.59-0.60 range — the larger
encoder anchor is a clear, consistent improvement in absolute terms, as
expected.

### Frozen control (Step 2)

Frozen DeBERTa-v3-large linear-probe anchor (single seed 42, mirroring
`scripts/train_frozen_text_foundation.py`'s one-time-anchor convention):
test weighted F1 **0.5521** (macro F1 0.3161) — also a clear improvement
over the RoBERTa-base frozen anchor's 0.4416.

Full residual stack on frozen features, 3 graph-stack seeds:

| seed | test weighted F1 | paired gain vs. frozen anchor | all 7 nonzero |
|---|---|---|---|
| 42 | 0.5732 | +0.0210 | No |
| 1337 | 0.5800 | +0.0278 | Yes |
| 2024 | 0.5714 | +0.0193 | No |
| **mean gain** | | **+0.0227** | |

### Fine-tuned residual stack (Step 3)

Full residual stack on the fine-tuned DeBERTa-v3-large foundation (seed-42
checkpoint's `text_ctx`/z_text cache, 3 graph-stack seeds), paired against
the **same-seed-numbered** fine-tuned anchor (this project's established
pairing convention — see `scripts/compute_diff_in_diff_deberta_large.py`'s
docstring):

| seed | test weighted F1 | paired gain vs. fine-tuned anchor (same seed) | all 7 nonzero |
|---|---|---|---|
| 42 | 0.6640 | +0.0044 | Yes |
| 1337 | 0.6634 | +0.0012 | Yes |
| 2024 | 0.6627 | +0.0060 | Yes |
| **mean gain** | | **+0.0039** | |

### Difference-in-differences (Step 4)

Same method as `docs/DIFF_IN_DIFF.md`: independent-groups percentile
bootstrap, **100,000 resamples**, fixed RNG seed 20260818.

| comparator | n (frozen/f.t.) | DiD | 95% CI | excludes zero |
|---|---|---|---|---|
| DeBERTa-v3-large, k=8 | 3/3 | **+0.0188** | **[+0.0144, +0.0239]** | Yes |

For reference, the RoBERTa-base primary result (`docs/DIFF_IN_DIFF.md`,
unchanged): DiD = +0.0444, 95% CI [+0.0347, +0.0560]. Same sign, same
"excludes zero" conclusion, smaller magnitude under the larger encoder.

## Table for Section IV / appendix insertion (not yet inserted)

```latex
\begin{table}[t]
\centering
\caption{Second-encoder replication (DeBERTa-v3-large): paired residual
gain (Full stack minus text-only anchor, test weighted F1) under frozen vs.
fine-tuned text representations, and the regime difference-in-differences
(independent-groups percentile bootstrap, $n{=}100{,}000$ resamples), 95\%
interval. RoBERTa-base row reproduced from Table~\ref{tab:diff-in-diff} for
reference, unchanged.}
\label{tab:diff-in-diff-deberta-large}
\begin{tabular}{lccc}
\toprule
encoder & frozen gain (n=3) & fine-tuned gain (n=3) & DiD [95\% CI] \\
\midrule
RoBERTa-base (Table~\ref{tab:diff-in-diff}) & +0.0356 avg. & $-$0.0088 avg. (n=7) & +0.0444 [+0.0347, +0.0560] \\
DeBERTa-v3-large & +0.0227 & +0.0039 & +0.0188 [+0.0144, +0.0239] \\
\bottomrule
\end{tabular}
\end{table}
```

## One-paragraph result (drafted for later Section IV / appendix insertion, not yet inserted)

> To test whether the fine-tuning boundary reflects RoBERTa-base's specific
> capacity rather than a general property of fine-tuned text representations,
> we pre-registered and ran the same frozen-vs-fine-tuned residual-gain
> comparison on `microsoft/deberta-v3-large` (24 layers, hidden 1024, ~3.5x
> RoBERTa-base's parameter count, a different attention architecture
> entirely). The frozen-regime gain remains substantial and the
> difference-in-differences remains positive and excludes zero
> (+0.0188, 95\% CI [+0.0144, +0.0239]), replicating the paper's core
> interaction claim. However, the fine-tuned-regime gain itself, while an
> order of magnitude smaller than the frozen-regime gain (+0.0039 vs.
> +0.0227), was small but consistently positive across all 3 seeds rather
> than null-to-negative as under RoBERTa-base — by our pre-registered
> reading rule this means the stronger claim ("the graph stack adds no
> value on any fine-tuned foundation") does not replicate unconditionally,
> and we narrow the paper's scope claim accordingly. We note this
> fine-tuned-regime result rests on only 3 seeds, an order of magnitude
> smaller effect than the frozen-regime one, and should be weighed with
> correspondingly less confidence than the paper's primary n=7 RoBERTa-base
> result.

## Compute / VRAM / wall-clock

- GPU: single NVIDIA RTX 5090 (32,607 MiB), bf16 autocast throughout,
  matching every other experiment in this repo.
- **Step 1 (LoRA fine-tuning of DeBERTa-v3-large, 3 seeds)**: peak VRAM
  observed via `nvidia-smi` during training was **~26.1 GB / 32.6 GB**
  (98% GPU utilization) — comfortably within this card's budget but likely
  **will not fit on a 24GB card** (e.g. RTX 3090/4090) without reducing
  batch size, enabling gradient checkpointing, or both; this was not tested
  since it wasn't necessary here. ~169s/epoch, up to 10 epochs, ≈1520-1690s
  (~25-28 min) wall-clock per seed, **≈82 min total for Step 1**.
- **Step 2 (frozen feature caching + linear probe + Full-stack graph
  training, 3 seeds)**: frozen feature extraction is a single forward pass
  over ~13.7k utterances, <15s per split; the linear probe itself is cheap
  (~1.5s/epoch, up to 77 epochs, ~115s); the 3 Full-stack graph-training
  runs are ~24s/epoch, 19-24 epochs each, ≈455-580s (~8-10 min) per seed.
- **Step 3 (text_ctx cache build + Full-stack graph training on the
  fine-tuned foundation, 3 seeds)**: cache build ≈46s total (train+dev+test);
  graph training ≈24s/epoch, 11-15 epochs, ≈267-360s (~5-6 min) per seed.
- **Total wall-clock, sum of all training-epoch time across every run in
  this replication (Steps 1-3): ≈7,523s ≈ 125 minutes (~2.1 hours)**, plus
  a few minutes of one-time cache-building passes — the whole replication,
  including the equality-at-init gate check, completed in a single session.
- Disk: new caches (`text_deberta_large`, `text_ctx_deberta_large`, and
  their `_logits` counterparts) add ~13.7k utterances x 1024-d float32 x 2
  cache variants ≈ negligible (~110 MB total); checkpoints for 3 fine-tuned
  LoRA foundations + 1 linear probe + 6 graph-stack runs add a few hundred
  MB more. No disk pressure encountered (408 GB free at start).

## New dependencies / code changes (full list, for audit)

- `sentencepiece`, `protobuf` added via `uv add` (DeBERTa-v3's tokenizer).
- `src/rapport/models/text_classifier.py`: `ContextTextClassifier` gained
  `lora_target_modules` (default unchanged) and explicit
  `torch_dtype=torch.float32` on load (deberta-v3-large ships natively in
  fp16); `add_pooling_layer` passed conditionally.
- `src/rapport/models/rapport_model.py`: `RapportModel` gained
  `text_feature_dim` (default 768, unchanged for every existing config).
- `src/rapport/models/backbones.py`: new `DebertaV2Backbone` (additive).
- `src/rapport/data/meld.py`: `MELDCachedDataset` gained `text_feature_dim`
  (default None -> FEATURE_DIM, unchanged behavior) so its per-feature shape
  assertion no longer hardcodes 768 for the text modality specifically.
- `scripts/train_rapport.py`: `--text_feature_dim` CLI/param passthrough
  (default 768, unchanged).
- New scripts (all additive, none modify an existing script's behavior):
  `train_deberta_large_text_anchor.py`, `build_text_ctx_cache_deberta_large.py`,
  `build_frozen_text_cache_deberta_large.py`,
  `train_frozen_deberta_large_foundation.py`,
  `run_deberta_large_residual_matrix.py`,
  `compute_diff_in_diff_deberta_large.py`.
- New test file `tests/test_rapport_model_residual_large_encoder.py`
  (equality-at-init at text_feature_dim=1024, plus a check that the
  default 768-d path is unperturbed). Full existing test suite (103 tests,
  non-slow) passes unchanged after every code change in this replication.

## Gate checklist

- [x] `docs/PREREG_DeBERTa-v3-large.md` committed and pushed before any
  DeBERTa-v3-large training run (commit `f5fcc3f`).
- [x] Equality-at-init verified on the large foundation's real cached data
  (both frozen and fine-tuned conditions) immediately before launching the
  residual-stack training runs (`scripts/run_deberta_large_residual_matrix.py`
  asserts this at startup; both PASS).
- [x] Verdict reported plainly against the pre-registered reading rule:
  **DOES NOT REPLICATE** (with the magnitude/n=3 caveat stated above).
- [x] No existing value, cache, or result changed.
