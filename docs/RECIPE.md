# RAPPORT locked training recipe

**STATUS: PERMANENTLY FROZEN** (as of the FINAL RECIPE LOCK phase,
`docs/DIAGNOSIS.md`). Every value below applies to **every** future
experiment config (`speaker_only`, `full`, `minus_*`, `*_lora`). **No
further recipe iterations, regardless of gate outcome on the locked-recipe
anchor run.** If a future gate fails under this recipe, the response is to
reassess the *architecture* (RAPPORT's relational/shift/temporal
components), not to reopen tuning of anything in this document. This file
is the historical record of how each value was chosen, kept for
provenance — it is not an invitation to keep adjusting them.

## Scope note — Phase T (contextual text encoder) is a documented exception

The locked recipe below governs the **GNN-based architectures over frozen,
cached V/A/T features** (`speaker_only`, `full`, `minus_*`, `*_lora`).
**Phase T** is a separate, text-only foundation-rebuild phase: a
context-window, LoRA-tuned RoBERTa trained end-to-end (no GNN, no fusion,
no frozen backbone), which is a different regime the locked recipe was
never calibrated against (different optimizer regime — from-scratch adapter
tuning at lr=2e-4 vs. the locked lr=1e-4; different batching — by utterance,
not by dialogue; no recurrent/graph state to even apply focal loss's
per-timestep alpha weighting against in the same way). This is a scope
boundary, not a reopening of the freeze: the table below still applies
unchanged to every GNN-based config once Phase T's encoder is integrated
into one.

### Phase T frozen recipe (final, as of the gate-failure diagnosis — `docs/PHASE_T_DIAGNOSIS.md` / `docs/PHASE_T_STEP4.md`)

**Training:** roberta-base + LoRA (r=8, alpha=16, dropout=0.05,
target=[query,value]), masked-mean pooling over `last_hidden_state`,
**plain cross-entropy loss** (`nn.CrossEntropyLoss`, no adjustment at train
time), AdamW lr=2e-4, linear warmup 10%, max 10 epochs, early stop patience
3 on val macro F1, bf16, batch by utterance (batch size 32), context k=8,
max_length=256.

**Inference:** post-hoc-only logit adjustment (Menon et al. 2021's *other*
variant), `logits - tau_eval*log(prior)`, **tau_eval=0.25** — selected once
on seed 42's val split (highest val macro F1 subject to all-7-nonzero AND
val weighted F1 >= 0.58), then frozen and applied identically to seeds
1337/2024, per this project's established tau-freezing convention
(Amendment 2 below).

**Frozen encoder checkpoint (Phase N4 onward):** the seed-42 run of this
recipe, `outputs/context_text_plain_ce_seed42/best_model.pt` (epoch 9, val
macro F1 0.4530), sha256
`447f369f02aad5297e7050a41f0ac6b0926bac70f70f467394293e4b11bb2f23`, is THE
project text encoder for all of Phase N4 — frozen, not retrained per seed
or per ablation config. See `docs/PHASE_N4.md` Step 0.

**Superseded:** the original Phase T recipe trained with logit adjustment
baked into the loss itself (`LogitAdjustedLoss`, tau=1.0, "CE + LA") and
used raw logits at eval with no post-hoc step. That recipe passed weighted
F1 (0.6041) but produced a systematic fear/disgust collapse (both exactly
0.0 test F1) that the diagnosis traced to the *training* loss, not the
representation or an implementation bug — see `docs/PHASE_T_DIAGNOSIS.md`.
`LogitAdjustedLoss` (`src/rapport/training/losses.py`) is kept in the
codebase (used by `scripts/train_context_text.py --loss la` for
regression/comparison purposes) but is **not** the frozen recipe.

## Locked recipe (final)

| component | value | source |
|---|---|---|
| optimizer | AdamW, lr=1e-4 | original speaker_only reproduction target |
| schedule | cosine annealing, T_max=max_epochs | " |
| loss | focal loss, gamma=3.0 | " |
| loss alpha | **tempered class-balanced, tau=0.5** (see below) | FINAL RECIPE LOCK tau-selection sweep |
| dropout | 0.5 | original |
| GNN hidden dim | 256 | original |
| max epochs | 100 | original |
| early stop | patience 10, **metric: val weighted F1** | original (audited twice, not changed — see below) |
| precision | bf16 autocast | original |
| BLAS thread caps | OMP/OPENBLAS/MKL_NUM_THREADS=8 | recalibration phase housekeeping |

## Amendment 2 (final) — tempered alpha, tau=0.5

**Adopted:** yes — the concluding amendment of the FINAL RECIPE LOCK phase.
Supersedes Amendment 1's tau=1.0 (full inverse frequency) below; the
formula generalizes to `w_c = (1/f_c)^tau`, so tau=1.0 (Amendment 1) and
tau=0.0 (original gamma-only) are both special cases of the same dial, not
separate mechanisms.

**Why:** Amendment 1 (tau=1.0) fixed the disgust/fear collapse but
overcorrected — neutral recall dropped enough to fail the (then-current)
accuracy gate in all 3 seeds. tau is the continuous dial between "tau=0:
disgust collapses to 0" and "tau=1: overcorrects, hurts the 48%-majority
class enough to fail accuracy." Selected once, on **val** metrics, **seed
42 only**, fixed criterion decided in advance: *highest val macro F1
subject to val fear F1 > 0 and val disgust F1 > 0* — never test metrics,
never other seeds, to keep the selection honest (a 1-seed, val-only search
over a single scalar is a small, bounded, pre-committed search, not a
tuning fishing expedition).

| tau | val weighted F1 | val macro F1 | val fear F1 | val disgust F1 | constraint |
|---|---|---|---|---|---|
| 0.00 | 0.5068 | 0.3275 | 0.049 | 0.000 | fails |
| 0.25 | 0.5122 | 0.3325 | 0.049 | 0.000 | fails |
| **0.50** | **0.5186** | **0.3648** | **0.157** | **0.067** | **passes — selected** |
| 1.00 | 0.4744 | 0.3415 | 0.195 | 0.051 | passes |

tau=0.5 both satisfies the constraint and has the highest val macro F1 of
the two constraint-satisfying candidates (0.3648 vs. tau=1.0's 0.3415) —
a clean win on the pre-registered criterion, not a close call requiring
judgment. Full sweep detail: `docs/DIAGNOSIS.md`, FINAL RECIPE LOCK
section. `focal_tau: 0.5` is now the permanent default in
`configs/training/default.yaml`.

## Amendment history

### Amendment 1 (superseded by Amendment 2 above) — class-balanced (inverse-frequency, tau=1.0) alpha for FocalLoss

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
