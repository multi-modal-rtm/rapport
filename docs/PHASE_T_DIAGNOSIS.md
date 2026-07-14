# PHASE T GATE-FAILURE DIAGNOSIS

Investigating why trained-time logit adjustment (LA) failed to rescue
fear/disgust in the seed-42 run (`docs/PHASE_T.md`: test weighted F1
0.6041, but fear F1 = disgust F1 = 0.000). **Diagnostics only — no
retraining until Step 4's proposal is signed off.** Steps 1-3 use the
existing `outputs/context_text_seed42/best_model.pt` checkpoint
unmodified. Numerical work: interactive audit (Step 1) +
`scripts/diagnose_logit_adjustment.py` (Steps 2-3, output also saved to
`outputs/logit_adjustment_diagnosis.json`).

---

## STEP 1 — LA implementation audit (numerical, not code-reading)

### (a) Sign — toy 3-class case, priors [0.7, 0.2, 0.1], tau=1.0

Hand-computed `adjusted_logits = logits - tau*log(prior)` against
`LogitAdjustedLoss`'s actual output, for 3 examples (one per class):

```
log_priors = [-0.3567, -1.6094, -2.3026]        # log([0.7, 0.2, 0.1])
raw logits  = [[2.0, 1.0, 0.5], [0.1, 3.0, -1.0], [0.0, 0.0, 5.0]]
adjusted    = [[2.357, 2.609, 2.803], [0.457, 4.609, 1.303], [0.357, 1.609, 7.303]]
hand-computed CE loss over `adjusted` = 0.467764
LogitAdjustedLoss(priors, tau=1.0)(logits, targets) = 0.467764   -- exact match
```

**Sign confirmed correct.** The additive boost `-tau*log(prior_c)` is
strictly larger for rarer classes (rare class idx=2, prior=0.1: **+2.303**;
common class idx=0, prior=0.7: **+0.357**) — this is Menon et al. 2021 Eq.
5's training-time formulation: the model must produce a raw-logit margin of
at least `tau*log(pi_y/pi_c)` over each competitor `c` to achieve zero
adjusted-loss on a true-rare-class example, i.e. it's cheaper (in training
loss) for a rare true class to "beat" a common competitor than the reverse
— exactly the intended correction, and the correct Menon variant per the
original phase instructions ("At eval, use raw logits" — i.e. the
loss-based/training-time variant, not post-hoc-only).

### (b) Priors from TRAIN split, normalized

`compute_class_priors(train_df["label"], 7)` on the real
`data/meld/processed/train.parquet`:

| class | prior | log(prior) | train-time boost (`-log(prior)`, tau=1) |
|---|---|---|---|
| neutral | 0.4715 | -0.7519 | +0.7519 |
| joy | 0.1745 | -1.7458 | +1.7458 |
| sadness | 0.0684 | -2.6826 | +2.6826 |
| anger | 0.1110 | -2.1979 | +2.1979 |
| surprise | 0.1206 | -2.1149 | +2.1149 |
| **fear** | **0.0268** | -3.6182 | **+3.6182** |
| **disgust** | **0.0271** | -3.6070 | **+3.6070** |

Sums to 1.0000. Matches the known published train-split proportions
(neutral ~47%, fear/disgust ~2.7% each — `docs/RECIPE.md`'s Amendment 1
table). Fear and disgust receive by far the largest training-time boost
(+3.6, roughly 5x neutral's), as intended.

### (c) Adjustment NOT applied in the eval path — verified numerically, not by reading the code

Loaded the real seed-42 checkpoint, ran one real dev batch through the
model, and compared three things directly:
1. `argmax(raw_logits)` computed ad hoc,
2. `argmax(raw_logits - 1.0*log_priors)` (what the *training* loss
   effectively pushes toward),
3. `train_context_text.evaluate_split()`'s actual returned predictions for
   the same batch.

(1) and (3) are **identical** (`[4, 4, 0, 4, 4, 0, 0, 1, ...]`), and (1)
differs from (2) on 3/32 examples in this batch — including one flip to
class 6 (disgust) under the adjusted-but-not-actually-used path. This
numerically proves `evaluate_split` uses raw logits, not the training-time
formula, exactly as designed. Additionally, re-running the full dev split
through the checkpoint reproduces the training run's logged epoch-4 numbers
exactly: recomputed weighted F1 0.5780 / macro F1 0.3800 vs. logged 0.5780
/ 0.3800.

### Step 1 conclusion: no implementation bug

All four checks pass. The sign is correct, priors are correctly computed
from the train split, the eval path is confirmed (numerically, via a live
checkpoint + dataloader, not just by reading `train_context_text.py`) to
use raw logits, and the actual log-prior vector used in the real run is
reported above. **LA is implemented correctly per Menon et al. 2021's
training-time (loss-based) variant.**

---

## STEP 2 — Prediction anatomy (test split, existing checkpoint)

`scripts/diagnose_logit_adjustment.py`, full 2610-utterance test split.

### Predicted-class histogram vs. true-class histogram

| class | true count | predicted count |
|---|---|---|
| neutral | 1256 | 1572 |
| joy | 402 | 352 |
| sadness | 208 | 55 |
| anger | 345 | 242 |
| surprise | 281 | 389 |
| **fear** | 50 | **0** |
| **disgust** | 68 | **0** |

Same total collapse pattern documented for the GNN architecture in
`docs/DIAGNOSIS.md`: fear and disgust are never predicted at all.

### Mean/median rank of the TRUE class within each example's own logit ordering (rank 1 = highest-scoring class)

| class | support | mean rank | median rank |
|---|---|---|---|
| neutral | 1256 | 1.14 | 1 |
| joy | 402 | 1.72 | 1 |
| surprise | 281 | 1.54 | 1 |
| anger | 345 | 2.13 | 2 |
| sadness | 208 | 3.12 | 3 |
| **fear** | 50 | **5.98** | **6** |
| **disgust** | 68 | **5.87** | **6** |

Full rank distributions:

```
fear:    {rank3: 2, rank4: 1, rank5: 6, rank6: 28, rank7: 13}   (82% at rank 6-7)
disgust: {rank3: 2, rank4: 3, rank5: 16, rank6: 28, rank7: 19}  (69% at rank 6-7)
```

### Step 2 conclusion: the pre-registered "rank 2-3" hypothesis is falsified — this is NOT a threshold problem

The phase instructions' own decision rule was explicit: *"If fear/disgust
are consistently rank 2-3 rather than rank 6-7, the signal exists and is
merely under-threshold (points to Step 3 succeeding)."* **The measured
result is the rank-6-7 case, not rank-2-3** — and it's not close: every
other class has mean rank ≤ 3.12, while fear (5.98) and disgust (5.87) sit
right at the bottom of a 7-way ordering, a qualitatively different regime.
For the substantial majority of true-fear and true-disgust test utterances,
the model's raw logits actively rank them as the *least* or *second-least*
likely class, not a near-miss runner-up. This is real evidence *against*
Step 3 succeeding, going in, not for it.

---

## STEP 3 — Post-hoc (inference-time-only) logit adjustment sweep, VAL

Applied `logits - tau_eval*log(prior)` **at inference only**, on top of the
already-LA-trained checkpoint (i.e. this stacks a second adjustment on top
of the one already baked into training — not the same as the "pure
post-hoc" Menon variant, which pairs post-hoc adjustment with *plain-CE*
training; see Step 4).

| tau_eval | val weighted F1 | val macro F1 | fear F1 | disgust F1 | all 7 nonzero? |
|---|---|---|---|---|---|
| 0.25 | 0.5895 | 0.4019 | 0.000 | 0.087 | no |
| 0.50 | 0.5890 | 0.4045 | 0.000 | 0.083 | no |
| 0.75 | 0.5881 | 0.4086 | 0.048 | 0.077 | **yes** |
| 1.00 | 0.5848 | 0.4109 | 0.091 | 0.065 | **yes** |

**No tau_eval simultaneously satisfies (all 7 nonzero AND weighted F1 ≥
0.59).** There's a real, monotonic trade-off, the same shape documented
repeatedly elsewhere in this project (`docs/RECIPE.md`'s tau-selection
sweep, the original class-balanced-alpha amendment): pushing tau_eval up
activates fear (0 → 0.091) and initially disgust, but disgust F1 itself
*peaks at tau_eval=0.25 (0.087) and monotonically declines* as tau_eval
increases further (0.087 → 0.083 → 0.077 → 0.065) — post-hoc
over-correction starts costing disgust before it's finished helping fear.
Weighted F1 declines monotonically across the whole sweep (0.5895 →
0.5848), consistent with the majority-class-recall cost documented
throughout this project whenever rare-class correction is strengthened.

**No qualifying tau_eval exists, so no TEST evaluation was run** (per the
script's pre-registered rule: TEST is only touched if a VAL candidate
clears both bars).

---

## STEP 4 — Proposed amendment (evidence-based, NOT applied — stopping for sign-off)

Steps 1-3 collectively rule out an implementation bug (Step 1) and rule out
"the signal is there, just needs a threshold nudge" (Steps 2-3 both point
the same direction: fear/disgust rank at the bottom of the model's own
logit ordering for the large majority of their true examples, and no
amount of post-hoc top-up on the current checkpoint clears both gate
criteria at once). This matches the phase instructions' own second
branch: **"the LA loss was correct but under-internalized."**

**Proposed amendment: switch to the *other* Menon et al. 2021 variant —
train with plain cross-entropy (no logit adjustment in the training loss),
then apply logit adjustment only at inference time (`logits -
tau_eval*log(prior)`, sweep tau_eval post-hoc exactly as Step 3 did, but
now against a plain-CE-trained checkpoint instead of stacking on top of an
already-LA-trained one).**

Rationale:
- Step 1 already confirms the *training-time* LA formula itself has no bug
  — so if it's "under-internalized," the fix is a different *training*
  regime, not a different formula.
- Stacking Step 3's post-hoc adjustment on top of an already-LA-trained
  model is a form of double-counting (the model's raw logits already
  reflect one round of margin-widening pressure from training; adding a
  second, unrelated additive correction on top has no principled
  coefficient — there's no reason `tau_train=1.0` plus `tau_eval=X` should
  compose cleanly). Plain-CE + post-hoc is the theoretically clean
  alternative: exactly one adjustment, applied once, with its own
  independently-tuned tau via the same val-only sweep procedure already
  built (`scripts/diagnose_logit_adjustment.py`'s `step3_posthoc_sweep`,
  reusable as-is against a new checkpoint).
- This is a bounded, one-shot amendment: one retrain (plain CE, otherwise
  identical recipe — same LoRA config, same lr/warmup/schedule/batching),
  then the same post-hoc tau sweep already implemented, on VAL only, same
  pre-registered thresholds (all 7 nonzero AND weighted F1 ≥ 0.59) before
  touching TEST.

**Stopping here for sign-off, per instructions — no retraining has been
performed.** `outputs/context_text_seed42/` (the LA-trained checkpoint) is
unmodified; nothing in `src/rapport/training/losses.py` or
`scripts/train_context_text.py` has changed as part of this diagnosis.
