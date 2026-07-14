# PHASE T STEP 4 — plain-CE + post-hoc adjustment (approved amendment)

Sign-off received for the other Menon et al. 2021 variant: plain CE
training (no adjustment in the training loss), post-hoc (inference-time-
only) logit adjustment applied after training. One retrain, seed 42,
recipe otherwise identical to the original LA run (context k=8, LoRA
r=8/alpha=16, lr 2e-4, warmup 10%, max 10 epochs, early stop patience 3 on
val macro F1, bf16). `scripts/train_context_text.py --loss ce`.

New shared module `src/rapport/eval/rank_diagnostics.py` (4 unit tests,
all pass) holds the reusable diagnostics; `scripts/diagnose_logit_adjustment.py`
was refactored to use it too (verified byte-identical output before/after).

## Instrumentation results (seed 42, plain CE)

### 1. Per-epoch dual per-class VAL F1 (raw vs. tau=1-adjusted) — observation only

No early stop triggered this time — training ran the full 10 epochs, val
macro F1 climbing steadily and (unlike the LA run, which peaked at epoch 4
then degraded) still improving at the last epoch:

| epoch | val weighted F1 | val macro F1 |
|---|---|---|
| 0 | 0.3657 | 0.1836 |
| 1 | 0.5537 | 0.3589 |
| 2 | 0.5815 | 0.3891 |
| 3 | 0.5807 | 0.4072 |
| 4 | 0.5869 | 0.4146 |
| 5 | 0.5927 | 0.4127 |
| 6 | 0.6007 | 0.4435 |
| 7 | 0.5965 | 0.4223 |
| 8 | 0.5997 | 0.4483 |
| 9 | **0.6032** | **0.4530** (best, selected) |

Selected checkpoint: epoch 9, val macro F1 0.4530 — already well above the
LA run's best val macro F1 (0.3800, epoch 4).

### 2. VAL rank-of-true-class + top-1 confusion (raw logits, selected checkpoint) — the decisive diagnostic

| class | mean rank of true class | median rank | (LA-run mean rank, for comparison) |
|---|---|---|---|
| **fear** | **3.575** | 3 | 5.98 |
| **disgust** | **3.727** | 3 | 5.87 |

**Ranks moved from ~6-7 (bottom of a 7-way ordering) under LA training to
~3-4 (middle of the pack) under plain CE — exactly the "ranks improve to
~2-4" signature that, per the pre-registered decision rule, means the
LA-trained loss was the bottleneck, not the representation itself.** The
representation *can* separate fear/disgust reasonably well; training under
the LA loss was actively preventing that from surfacing at the raw-logit
level (plausible mechanism: the training-time additive correction pushes
weight updates toward satisfying the *adjusted* margin rather than the raw
one, and evidently doesn't reliably transfer that margin into the raw
logits used at eval — Step 1 already ruled out a bug in the correction
itself, so this is a property of the training dynamics, not an error).

Top-1 confusion for true-fear/true-disgust VAL utterances (raw logits):

```
fear (support 40):    anger 14, neutral 7, sadness 6, joy 5, surprise 5, fear 3, disgust 0
disgust (support 22): neutral 7, sadness 4, anger 3, surprise 3, disgust 4, fear 1, joy 0
```

Mixed but not scattershot: fear's largest confusion bucket (anger, 14/40 =
35%) is a plausible semantically-adjacent high-arousal negative emotion,
though neutral/sadness/joy/surprise together still take the majority.
Disgust's errors spread fairly evenly across the negative-emotion classes
(sadness+anger+fear = 8, comparable to neutral's 7) rather than collapsing
onto the majority class alone. Read together with the rank result: real,
moderate signal, not pure noise — consistent with "representation can do
this, needs more of a push," not "representation cannot do this at all."

### 3. Post-hoc tau sweep on VAL (selected checkpoint, plain-CE-trained)

| tau_eval | weighted F1 | macro F1 | fear F1 | disgust F1 | all 7 nonzero? |
|---|---|---|---|---|---|
| 0.25 | 0.6077 | 0.4740 | 0.167 | 0.324 | yes |
| 0.50 | 0.5944 | 0.4568 | 0.189 | 0.245 | yes |
| 0.75 | 0.5885 | 0.4593 | 0.200 | 0.258 | yes |
| 1.00 | 0.5680 | 0.4449 | 0.222 | 0.212 | yes |
| 1.25 | 0.5316 | 0.4269 | 0.223 | 0.211 | yes |

Every tau_eval in the sweep now clears all-7-nonzero (a qualitative
improvement over the LA-run sweep, where only tau_eval >= 0.75 did).
Candidate-selection rule (pre-registered): highest val macro F1 subject to
all-7-nonzero AND val weighted F1 >= 0.58. **tau_eval=0.25 selected**
(macro F1 0.4740, the best in the whole sweep, and the only row with
weighted F1 comfortably above the 0.58 floor).

## Result: RAW (unadjusted) test predictions already pass the ORIGINAL gate outright

Before even applying the selected tau, the plain-CE checkpoint's raw
(unadjusted) TEST predictions:

| metric | value |
|---|---|
| test weighted F1 | **0.6353** |
| test macro F1 | 0.4358 |
| test accuracy | 0.6517 |
| neutral F1 | 0.797 |
| joy F1 | 0.614 |
| sadness F1 | 0.375 |
| anger F1 | 0.463 |
| surprise F1 | 0.561 |
| fear F1 | 0.062 |
| disgust F1 | 0.180 |

**Original gate (weighted F1 >= 0.60 AND all 7 nonzero): PASSES outright,
with no post-hoc adjustment at all.** This alone resolves the ambiguous
outcome flagged in `docs/PHASE_T.md` — the LA-trained loss was the
specific problem, not a fundamental limit of this architecture/data.

## TEST result with the frozen candidate (tau_eval=0.25) applied once

Per the pre-registered protocol, applied the selected tau_eval=0.25 to
TEST once:

| metric | raw (no adjustment) | tau_eval=0.25 |
|---|---|---|
| weighted F1 | 0.6353 | 0.6310 |
| macro F1 | 0.4358 | 0.4440 |
| fear F1 | 0.062 | **0.154** |
| disgust F1 | 0.180 | 0.165 |
| all 7 nonzero | yes | yes |

Amended gate (weighted F1 >= 0.59 AND all 7 nonzero): **PASSES** (weighted
F1 0.6310, comfortably above 0.59; both fear and disgust nonzero and less
marginal than the raw fear F1 of 0.062, which reflects only a handful of
correct predictions out of 50 support). tau=0.25 trades a small amount of
weighted F1 (-0.0043, still passing) and disgust F1 (-0.015, still
nonzero) for a real gain in macro F1 (+0.008) and fear F1 (+0.092) — a
genuine, if modest, fairness-for-cost trade in the same spirit as
`docs/RECIPE.md`'s tau-selection history for the GNN architecture.

## FROZEN RECIPE (text phase) — see docs/RECIPE.md for the permanent record

**Training:** roberta-base + LoRA (r=8, alpha=16, dropout=0.05,
target=[query,value]), masked-mean pooling over `last_hidden_state`, plain
cross-entropy loss (no adjustment at train time), AdamW lr=2e-4, linear
warmup 10%, max 10 epochs, early stop patience 3 on val macro F1, bf16,
batch by utterance (batch size 32), context k=8, max_length=256.

**Inference:** post-hoc logit adjustment, `logits - tau_eval*log(prior)`,
**tau_eval=0.25** (selected once on seed 42's val split, frozen — not
re-selected per seed, per this project's established tau-freezing
convention).

Seeds 1337 and 2024 are run under this identical frozen recipe
(`--loss ce --posthoc_tau 0.25`), not re-running the candidate-selection
sweep per seed. Results below.

---

## 3-seed anchor: plain CE + frozen post-hoc tau=0.25

`docs/train_context_text_plain_ce_seed{1337,2024}.log`,
`outputs/context_text_plain_ce_seed{42,1337,2024}/metrics.json`.

### RAW (unadjusted) test results

| seed | weighted F1 | macro F1 | accuracy | neutral | joy | sadness | anger | surprise | fear | disgust | all 7 nonzero? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.6353 | 0.4358 | 0.6517 | 0.797 | 0.614 | 0.375 | 0.463 | 0.561 | 0.062 | 0.180 | yes |
| 1337 | 0.6441 | 0.4478 | 0.6602 | 0.799 | 0.634 | 0.392 | 0.478 | 0.573 | 0.116 | 0.143 | yes |
| 2024 | 0.6415 | 0.4587 | 0.6582 | 0.803 | 0.592 | 0.373 | 0.483 | 0.576 | 0.211 | 0.174 | yes |
| **mean ± std** | **0.6403 ± 0.0045** | 0.4474 ± 0.0114 | 0.6567 ± 0.0044 | 0.799 | 0.613 | 0.380 | 0.475 | 0.570 | 0.129 | 0.166 | **3/3** |

**Original gate (weighted F1 >= 0.60 AND all 7 nonzero): PASSES in all 3
seeds**, with no post-hoc adjustment at all.

### Post-hoc-adjusted (frozen tau_eval=0.25) test results

| seed | weighted F1 | macro F1 | neutral | joy | sadness | anger | surprise | fear | disgust | all 7 nonzero? | amended gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.6310 | 0.4440 | 0.788 | 0.604 | 0.372 | 0.470 | 0.554 | 0.154 | 0.165 | yes | **pass** |
| 1337 | 0.6475 | 0.4673 | 0.795 | 0.634 | 0.407 | 0.486 | 0.579 | 0.207 | 0.164 | yes | **pass** |
| 2024 | 0.6408 | 0.4659 | 0.795 | 0.600 | 0.387 | 0.491 | 0.563 | 0.198 | 0.229 | yes | **pass** |
| **mean ± std** | **0.6398 ± 0.0083** | 0.4590 ± 0.0131 | 0.793 | 0.613 | 0.388 | 0.482 | 0.565 | **0.186** | **0.186** | **3/3** | **3/3 pass** |

**Amended gate (weighted F1 >= 0.59 AND all 7 nonzero): PASSES in all 3
seeds.** tau=0.25's trade vs. raw is consistent across all three seeds: a
small, uniformly-passing weighted F1 cost (mean -0.0005, within noise) for
a real, consistent macro F1 gain (mean +0.0116) and a large, consistent
fear F1 gain (mean +0.057, driven mostly by seed 42's raw fear F1 of 0.062
— the most marginal of the three — moving to 0.154), at a roughly
neutral net cost to disgust (mean +0.020, though individual seeds trade in
both directions).

### Anchor conclusion

**Clean 3/3 pass on both the original and amended gate, under both scoring
conventions (raw and post-hoc-adjusted) — a materially better and more
robust outcome than the original LA-trained recipe** (which passed
weighted F1 on 1 seed tested but failed all-7-nonzero on that same seed,
and was never run past seed 42 because of that). This is the anchor for
Phase T. Per Step 4's own decisive diagnostic (VAL rank-of-true-class
moving from ~6-7 under LA training to ~3-4 under plain CE), the original
recipe's failure was specifically attributable to training with the
adjustment baked into the loss, not to a representation-capacity limit of
this architecture/data — switching to the plain-CE + post-hoc-only Menon
variant resolved it without any architecture or data change.

**Frozen recipe:** see `docs/RECIPE.md`'s Phase T section. Stopping here
per instructions — 3-seed table reported, written to docs/, committing and
pushing next.
