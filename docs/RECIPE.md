# RAPPORT locked training recipe

This is the single source of truth for hyperparameters/design choices that
apply to **every** future experiment config (`speaker_only`, `full`,
`minus_*`, `*_lora`), not just the one they were first validated on. Changes
here require the same bounded-investigation-plus-sign-off process used to
add the first amendment below — this is not a place for casual tuning.

## Locked recipe (as of `speaker_only` gate-failure investigation)

| component | value | source |
|---|---|---|
| optimizer | AdamW, lr=1e-4 | original speaker_only reproduction target |
| schedule | cosine annealing, T_max=max_epochs | " |
| loss | focal loss, gamma=3.0 | " |
| loss alpha | **inverse-frequency, class-balanced** (see below) | amendment, this document |
| dropout | 0.5 | original |
| GNN hidden dim | 256 | original |
| max epochs | 100 | original |
| early stop | patience 10, **metric: val weighted F1** | original (audited, not changed — see below) |
| precision | bf16 autocast | original |
| BLAS thread caps | OMP/OPENBLAS/MKL_NUM_THREADS=8 | recalibration phase housekeeping |

## Amendment history

### Amendment 1 — class-balanced (inverse-frequency) alpha for FocalLoss

**Adopted:** yes, per explicit sign-off during the `speaker_only` N3
gate-failure investigation (`docs/DIAGNOSIS.md`, GATE-FAILURE INVESTIGATION
section).

**Problem it targets:** disgust F1 was exactly 0.0000 in all 3 seeds
(42/1337/2024) of the unamended recipe — not checkpoint-selection noise
(disgust's val F1 was a flat 0.0 line for the *entire* training run in
every seed, audited directly) and not a focal-loss implementation bug
(audited clean against Lin et al. 2017). Disgust (68/2610 test, ~2.6%) and
fear (50/2610, ~1.9%) are MELD's two rarest classes by a wide margin.
Gamma-only focal loss reweights easy-vs-hard examples but applies **no**
class-frequency correction — class-balanced (inverse-frequency) alpha is
the standard remedy for exactly this gap.

**What it does:** `FocalLoss` now accepts a per-class `alpha` tensor,
`compute_inverse_frequency_alpha` (`src/rapport/training/losses.py`)
computes it from the *train* split's label distribution:
`alpha_c = inv_freq_c * (num_classes / sum(inv_freq))`,
`inv_freq_c = total / (num_classes * count_c)` — normalized to mean 1, so
the loss's overall scale is unchanged, only its *distribution* across
classes shifts.

| class | train count | alpha |
|---|---|---|
| neutral | 4709 | 0.1304 |
| joy | 1743 | 0.3522 |
| sadness | 683 | 0.8988 |
| anger | 1109 | 0.5535 |
| surprise | 1205 | 0.5094 |
| fear | 268 | 2.2905 |
| disgust | 271 | 2.2652 |

**Measured effect (3-seed, speaker_only) — a real trade-off, not a clean win:**

| metric | pre-amendment (mean) | post-amendment (mean) | delta |
|---|---|---|---|
| test weighted F1 | 0.5353 | 0.4816 | **-0.0537** |
| test accuracy | 0.5763 | 0.4516 | **-0.1247** |
| test macro F1 | 0.3128 | 0.3179 | +0.0051 |
| fear F1 | 0.038 | 0.110 | **+0.072** |
| disgust F1 | **0.000** | **0.062** | **+0.062** |

**The amendment does exactly what it was designed to do** — disgust goes
from a total, systematic collapse (0/2610 predictions, every seed) to
genuine nonzero recall in every seed, and fear roughly triples. **It also
costs more than it gives back against this project's specific gate**:
weighted F1 and accuracy both drop substantially (accuracy now *below* the
constant-baseline gate in all 3 seeds, which the unamended recipe passed
cleanly), because down-weighting neutral (alpha=0.13) measurably hurts the
model's neutral recall (0.86 -> ~0.57 mean), and neutral is 48% of the test
set. Macro F1 (which weights all 7 classes equally) barely moves, because
the macro-F1 gain from fixing two tiny classes is almost exactly offset by
the loss on the majority class within that same equal-weighted average.

**Net gate outcome: still fails, on more criteria than before** (now fails
weighted-F1-floor *and* accuracy-baseline, vs. only weighted-F1-floor and
disgust-nonzero pre-amendment). See `docs/DIAGNOSIS.md` for full per-seed
numbers and interpretation. **This amendment stays adopted** (it was
signed off, it does what it was designed for, and macro F1 — arguably the
fairer aggregate metric for a 7-class problem with a 2.6%-smallest-class —
is flat-to-slightly-up) but the underlying gate-failure is not resolved by
it alone; per-seed detail is preserved in
`outputs/speaker_only_seed{42,1337,2024}_pre_alpha/` (pre-amendment
artifacts) vs. `outputs/speaker_only_seed{42,1337,2024}/` (post-amendment,
current).

### Early-stop / checkpoint-selection metric — audited, NOT changed

The gate-failure investigation's Step 2 explicitly tested whether
selecting the checkpoint by val weighted F1 (rather than val macro F1) was
hiding a better disgust/fear checkpoint earlier or later in training. It
was not: disgust's val F1 was 0.0 at *every* epoch in every seed (not just
the selected one), so there was no better checkpoint to select in the
first place. **Checkpoint selection and early stopping remain on val
weighted F1** — switching to macro F1 was evaluated and correctly not
adopted, since the evidence didn't support it as the mechanism.
