# RAPPORT Pipeline Diagnosis

Investigating the `speaker_only` (pre-pivot name: `A_baseline`) seed-42 run
that landed at test weighted F1 0.3765, far below the [0.60, 0.64]
reproduction target. Working hypothesis per the pivot: this exposed a
pipeline bug that would corrupt every future model equally, so it's being
fixed before any RAPPORT model work starts. **Diagnostics only — no training
code changes until Step 6 findings are reviewed and signed off.**

Evidence base: `outputs/A_baseline_seed42/` (config, checkpoint, classification
report, confusion matrix, TensorBoard log) and
`docs/DISCREPANCY_REPORT_A_seed42.md` from the prior phase.

---

## STEP 1 — Interpreting the headline numbers

Recomputed precisely from `outputs/A_baseline_seed42/classification_report.json`:

| Metric | Value |
|---|---|
| Test accuracy | **0.4295** |
| All-neutral constant-baseline accuracy | **0.4812** (1256/2610) |
| Test macro F1 | **0.1840** |
| Uniform-random-guess macro F1 (7 classes) | ~0.14 (rough reference point) |

**The model's accuracy (0.4295) is below the trivial all-neutral baseline
(0.4812).** This is the single most important fact in this diagnosis — it
is a qualitatively different (and much more alarming) failure than "the
model is weak." A model that learned *nothing at all* but still defaulted
toward the majority class would beat this. Getting a full ~5.5 points
*below* that floor requires the model to actively mispredict a meaningful
number of true-neutral utterances as something else, or to spread
predictions in a way that's actively anti-correlated with the labels on
some slice of the data — not just "fail to learn the minority classes."

This is corroborated by the confusion matrix (see prior discrepancy
report): true-neutral recall is only 74.7% (938/1256), with real,
non-trivial mass going to `sadness` as a second catch-all bucket that
absorbs misclassifications from classes with no particular semantic
similarity to sadness (212 from true-neutral, 64 from true-joy, 55 from
true-anger, 50 from true-surprise). A model that had simply failed to learn
would collapse cleanly onto the majority class (≈100% neutral recall,
0% everywhere else) and *match* the baseline, not undershoot it.

Macro F1 of 0.18 is only marginally above the ~0.14 you'd expect from
uniform random guessing across 7 classes, i.e. the model's per-class
behavior is close to noise once you don't credit it for the majority class.

### Consistent with

- **Feature/label misalignment** — features for utterance X paired with the
  label for a different utterance Y. This directly explains sub-baseline
  accuracy: the model would be fitting real signal to scrambled targets,
  which is actively worse than fitting no signal (predicting the
  unconditional mode) once the scrambling rate is high enough.
- **Stale cache** — if training pulled from a cache state that predates the
  split-key-collision fix (mixed dialogue_id namespaces across
  train/dev/test), or a fix's Python objects weren't cleanly reset, a
  meaningful fraction of cached features would belong to the wrong
  utterance. Same downstream effect as misalignment above.
- **Off-by-one in the dialogue unroll** — if the model's GRU/GAT hidden
  state at step t (built from utterance t's fused features) gets scored
  against `labels[t-1]` or `labels[t+1]` instead of `labels[t]`, every
  training and eval signal is shifted by one utterance within each
  dialogue. Same corrupting effect as misalignment, and notably would
  *not* show up as a shape/crash bug — training would proceed smoothly
  (as observed: loss decreased monotonically for 40 epochs) while learning
  a systematically wrong mapping.
- **Padding leaking into loss/metrics** — if padded positions (label = -1
  sentinel in the collate output) aren't correctly excluded and get treated
  as a real class index, or contribute to the metric's denominator/
  numerator incorrectly, both training gradients and reported numbers would
  be corrupted in a way that doesn't require the model to be "bad" at the
  task at all.

### NOT consistent with

- **Minority-class collapse alone.** Collapsing onto majority classes is a
  self-limiting failure mode — it caps out at *matching* the constant
  baseline, not beating it downward. Our own confusion matrix shows the
  model isn't even a clean majority-collapse (neutral recall 74.7%, not
  ~100%), so "the model gave up on rare classes" cannot by itself explain
  going below 0.4812 accuracy.
- **Mildly degraded features** (e.g. a somewhat-worse-than-ideal frozen
  backbone, or the suspected uninitialized RoBERTa pooler making text
  features noisier than intended). Noisy-but-real features degrade
  performance smoothly and land you *above* the constant baseline with
  informative errors concentrated between semantically similar classes —
  not below it. The magnitude and shape of this failure (sub-baseline
  accuracy, near-random macro F1, a semantically arbitrary second
  catch-all class) point at a structural pipeline bug actively corrupting
  the learning signal, not merely weakened inputs. (This doesn't rule out
  the RoBERTa pooler issue as *a* contributing problem — Step 3 tests that
  directly — just that it can't be the *whole* story here.)

**Next:** Step 2 audits cache integrity directly, since "stale/corrupted
cache" is both plausible and cheap to rule in or out first.

---

## STEP 2 — Cache integrity audit

### Was the cache used for training built after the collision fix?

Timestamp evidence (before touching anything): the cache's newest file
(`text_tokens/test/dia38_utt4.pt`) was written **08:30:48**; the training
run's Hydra config (marking run start) was written **08:32:10** — training
started ~82s after the cache finished, consistent with using the
post-collision-fix cache. But per instructions, "any doubt" means rebuild —
so it was rebuilt from empty regardless of this evidence.

**Rebuild from empty**, recorded before/after:

| | before delete | after full rebuild |
|---|---|---|
| train | 9988 (×6 subdirs) | 9988 (×6 subdirs) |
| dev | 1108 (×6 subdirs) | 1108 (×6 subdirs) |
| test | 2610 (×6 subdirs) | 2610 (×6 subdirs) |
| total files | 82,236 | 82,236 |
| disk | 19G | 19G |

Identical counts before/after — the pre-existing cache was already
structurally complete post-fix. (The rebuild did hit the same CUDA
allocator OOM near the tail of the `test` split as the original build; the
resumable/retry logic in `build_feature_cache.py` handled it the same way —
noted for awareness, not a new finding.)

### Count check (`scripts/audit_cache.py`)

Cache file count == index parquet row count for all 3 splits × all 3
pooled modalities (9 checks, all `[OK]`). No missing, no extra files.

### Cross-split "key overlap" — false alarm, documented and dismissed

The audit script's naive `dia{X}_utt{Y}` string-key overlap check reported
large "collisions" (train∩dev: 675, train∩test: 1739, dev∩test: 669). **This
is not a real bug.** MELD's `dialogue_id` restarts from 0 independently in
each split (confirmed in the prior phase: train 0–1038, dev 0–113, test
0–279), so the *raw* key `dia0_utt0` naturally exists in all three splits as
three different real utterances. The count check above already confirms the
actual on-disk cache files are correctly namespaced under
`cache/{modality}/{split}/...`, which is what actually prevents collisions
(fixed in the prior phase). Flagging this explicitly so it isn't mistaken
for a finding — the check as literally specified ("zero key overlap across
splits") only makes sense against split-qualified keys, which do have zero
overlap by construction of the directory layout.

### Live recompute vs. cache — REAL BUG FOUND

30 utterances sampled across all 3 splits; V/A/T pooled features recomputed
live through fresh backbone instances and compared to the cached tensors
(`atol=1e-3, rtol=1e-3`):

| modality | allclose failures | max abs diff |
|---|---|---|
| video | 0 / 30 | 0.000001 (fp noise) |
| audio | 11 / 30 | 1.706103 |
| **text** | **30 / 30** | **1.138753** |

**Video is clean** — MViTv2 is fully deterministic (frozen, eval mode, no
sequence-dependent padding since every clip is a fixed 8→16 frames), and
live vs. cached match to floating-point noise.

**Text is comprehensively broken — root cause confirmed directly.**
`RobertaModel.from_pretrained("roberta-base")` loads *without* pretrained
pooler weights (the "MISSING" load report seen when the backbone is first
built, flagged as a suspicion in the prior discrepancy report). Verified
directly just now: instantiating `RobertaBackbone()` twice with identical
input yields **different pooler weights** and pooled outputs differing by
up to 0.93. **The pooler is randomly re-initialized on every single
process/instantiation, with no seed control.** This means:
- The cached text features are a frozen snapshot of *one arbitrary random
  projection* of the [CLS] token from whichever process happened to build
  the cache — not a meaningful, reproducible representation of anything.
- Rebuilding the cache (as done earlier in this step) silently produces a
  **completely different** set of text features each time, with no way to
  detect this from file counts or shapes alone.
- This isn't "degraded" text signal — it's not a signal at all, just a
  fixed random hash of the [CLS] vector. This is a strong, direct,
  independent confirmation of the #1 suspect from the prior discrepancy
  report, now upgraded from "suspicion" to "root cause with reproduction."

**Audio is partially affected — root cause confirmed, different mechanism.**
Isolated the worst offender (`test/dia51_utt1`, diff 1.71): this clip is
**0.21 seconds long** (3413 samples), the kind of very-short outlier that
gets padded most heavily relative to its own length inside a batch.
Reproduced directly: computing this clip's feature alone vs. batched
alongside a much longer clip (forcing heavy padding) changes the pooled
output by up to 2.02. **Wav2Vec2's convolutional feature encoder is applied
to the padded waveform before any attention-mask-based exclusion happens**
(the mask only cleanly excludes padded *time steps* at the pooling step,
after the CNN has already convolved over them) — so the raw padding zeros
leak into the real audio's features near the padding boundary, more
severely for short clips. This is **batch-composition-dependent**: the same
utterance's cached feature depends on which other utterances happened to
land in its batch during the one-time cache build, not just on its own
audio. This is a real, reproducible bug — of a different character than the
text one (this is representational leakage between batch neighbors, not
outright randomness), but still means the audio cache is not a pure
function of each utterance in isolation.

### Step 2 conclusion

Cache *structure* (counts, split-namespacing, no missing/duplicate files)
is sound. Cache *content* is not: **text features are dominated by
per-build random noise (severe, affects 100% of samples), and audio
features carry a smaller batch-composition-dependent padding artifact
(affects a minority of samples, worse for short clips).** Video is clean.
This alone is sufficient to explain a training collapse — a downstream
model trained against a text modality that's pure noise, plus an audio
modality with batch-order-dependent corruption, would struggle to learn
*any* stable mapping, consistent with Step 1's sub-baseline-accuracy
finding.

**Next:** Step 3 quantifies this with per-modality linear probes, which
should show text and audio probes badly underperforming their healthy
ranges while video probes should look normal.

---

## STEP 3 — Per-modality linear probes

`scripts/probe_features.py`: sklearn `LogisticRegression(class_weight='balanced')`,
utterance-level (no dialogue context, no GNN), fit on train split cached
features, evaluated on test split. One probe per modality + one on
concatenated 3×768.

| modality | weighted F1 | healthy range | verdict |
|---|---|---|---|
| video | **0.1929** | 0.33–0.40 | **below range** |
| audio | **0.2870** | 0.38–0.45 | **below range** |
| text | **0.4019** | ≥ 0.53 | **below range** |
| concat | **0.3815** | ≥ text (0.40) | **fails — concat < text alone** |

(Re-ran with `StandardScaler` on all three unimodal probes to rule out a
raw-feature-scale artifact: 0.2025 / 0.2810 / 0.4132 respectively — nearly
identical, so this is a real feature-quality signal, not a probe
methodology gap. Also sanity-checked the data loading itself: train label
distribution matches the known published counts exactly, and video features
are non-degenerate (std ≈ 0.099 across dims, range roughly [-0.76, 1.0]) —
so the low scores aren't an artifact of a broken probe script either.)

**All three unimodal probes are below their healthy range — not just
text.** Per the branching in the task: text (0.40) is below the 0.45
threshold, so **the RoBERTa pooler hypothesis is material** — directly
corroborating Step 2's reproduction of the random-pooler bug. That said,
this is not a "text-only problem downstream of an otherwise-healthy
pipeline": video and audio are *also* below range, and **concat
underperforming text alone (0.3815 < 0.4019) is itself a red flag** — a
linear probe with L2 regularization should be able to approximately ignore
uninformative extra dimensions rather than get actively worse from them;
concat losing to its best constituent modality suggests the video/audio
noise isn't just "unhelpful," it's actively interfering.

Per-modality interpretation:
- **Text (0.40, below 0.53 floor):** consistent with and now doubly
  confirmed by Step 2's direct reproduction — the pooler is a fresh random
  projection per process, so this probe is measuring "how separable are
  classes under one arbitrary random hash of [CLS]," not "how separable are
  classes under RoBERTa's learned representation." That a random projection
  still gets to 0.40 (well above the ~0.14 uniform-chance floor for 7
  classes) is itself informative — [CLS] hidden states are rich enough that
  even a random linear+tanh projection retains partial class structure, but
  substantially less than the pretrained/proper pooling would.
- **Audio (0.29, below 0.38–0.45 range):** consistent with Step 2's
  batch-composition padding-leakage finding, though that finding only
  directly implicated 11/30 sampled utterances (37%) with typically small
  diffs (a few outliers larger) — plausible partial explanation, not
  necessarily the *whole* gap. Worth more attention in a later phase.
- **Video (0.19, below 0.33–0.40 range) — the single worst probe, despite
  passing Step 2's integrity check with zero corruption.** This means the
  video problem is not a caching bug: the cache is a faithful, exact record
  of what `MViTv2Backbone` actually computes. The most likely explanation,
  flagged as hypothesis #2 in the prior discrepancy report: our
  preprocessing stores 8 uniformly-sampled frames per clip, but MViTv2's
  Kinetics-pretrained positional encoding requires 16-frame input, so each
  frame is naively duplicated (`repeat_interleave`) to reach 16. This gives
  the model literally zero new temporal information beyond what's in the
  original 8 frames, and may also feed the positional encoding a motion
  pattern (each frame appearing twice in a row) unlike anything in its
  Kinetics pretraining distribution. This is a *feature-quality* problem,
  distinct in kind from the text/audio *caching* bugs — it would need a
  preprocessing change (re-extracting 16 native frames) to fix, not a
  backbone-wrapper fix.

### Step 3 conclusion

Per the task's explicit branch: text < 0.45 confirmed, pooler fix is
material and should be prepared (not applied) — **use
`last_hidden_state[:, 0]` (raw pretrained [CLS] token) or mean-over-tokens
instead of the randomly-initialized `pooler_output`.** But this step also
surfaces two *additional* problems beyond that single branch: an audio
padding-leakage bug (Step 2) and a video feature-quality problem stemming
from frame duplication (new finding, this step) — neither of which is "the
downstream model/GNN" and neither of which is fixed by touching the text
pooler. All three modalities need attention before this is a healthy
feature stack.

**Next:** Step 4 spot-checks label/feature alignment directly on
unambiguous examples, to rule in/out misalignment as an *additional*,
independent bug on top of the feature-quality issues found so far.

---

## STEP 4 — Label/feature alignment spot-check

Searched test-split transcripts for unambiguous joy/anger/surprise
exclamations (86 candidates found via keyword search; picked 10 spanning
all three emotions, including short one-liners like `"Yeah, get out!"` /
`"Woooo hoooo!!!"` where there's no reasonable alternative label). For each,
cross-checked: processed-index text/emotion/speaker vs. the original raw
`test_sent_emo.csv` row for the same `(dialogue_id, utterance_id)`, plus
confirmed the cache file path (`cache/text/test/dia{X}_utt{Y}.pt`) exists
and its filename embeds the matching `dia{X}_utt{Y}` key.

| dialogue_id | utterance_id | emotion | text | match |
|---|---|---|---|---|
| 132 | 4 | joy | "Oh yay! Great! ..." | ✓ |
| 233 | 10 | joy | "Yay, okay!" | ✓ |
| 216 | 0 | joy | "...I'm so happy for you!" | ✓ |
| 265 | 1 | joy | "...fantastic?" | ✓ |
| 83 | 4 | joy | "Woooo hoooo!!!" | ✓ |
| 32 | 0 | surprise | "Yes!! Yes! Yes! Yes!!..." | ✓ |
| 225 | 1 | joy | "Yes!! Yes!! I'm the next caller!..." | ✓ |
| 147 | 3 | anger | "Yeah, get out!" | ✓ |
| 211 | 0 | anger | "You, get out of my shop!" | ✓ |
| 211 | 5 | anger | "That's my wife!!! Get out!" | ✓ |

**10/10 clean.** Text, emotion label, and speaker all match exactly between
the processed index and the original raw CSV for every sample, and every
cache file resolves to the filename-matching utterance. No mismatch found.

### Step 4 conclusion

**Rules out** feature/label misalignment *at the CSV → processed-index →
cache-file-path level* — that whole chain is correctly wired. This does
**not** rule out misalignment happening *inside the dialogue unroll* (e.g.
scoring the GRU state built from utterance t's features against
`labels[t-1]` or `labels[t+1]`), which is a different bug class living in
`SocialGNN.forward`/`evaluate_split`, not in the data pipeline. That's
exactly what Step 5 is designed to catch.

**Next:** Step 5 — tiny-overfit test, plus direct checks of unroll
ordering, padding exclusion from loss/metrics, and graph-state isolation
across dialogues/batch elements.

---

## STEP 5 — Tiny-overfit test + model/collate/masking/state checks

`scripts/diagnose_overfit.py`, fresh `SocialGNN`, 5 real cached train
dialogues (lengths 14/7/13/9/15 = 58 real utterances + 17 padded slots in
one batch), AdamW lr=3e-3 (bumped from the trainer's 1e-4 purely to
converge within 300 steps for this diagnostic — no trainer code touched),
FocalLoss gamma=3.

### Tiny-overfit — PASS

```
step 000: loss=1.2032 train_acc=0.2414
step 050: loss=0.0081 train_acc=1.0000
...
step 299: loss=0.0001 train_acc=1.0000
FINAL train accuracy: 1.0000 over 58 unmasked utterances
```

Reaches 100% masked train accuracy by step 50 and holds it. The
model/optimizer/loss can fit this data perfectly given enough steps — rules
out "gradients don't flow" / "model is structurally incapable of learning"
as an explanation for the real run's failure.

### Padding exclusion — PASS (one sub-check redone; initial version was testing the wrong thing)

Three checks, using the already-overfit model:

1. **Collate correctness (direct inspection):** `batch["labels"]` at every
   padded position is exactly `-1`; at every real position it's in `0..6`.
   Confirmed directly on the real 5-dialogue batch. `collate_dialogues`'s
   `_pad_stack(..., pad_value=-1)` for labels is doing its job.
2. **Feature corruption at padded positions doesn't touch real-position
   outputs:** replaced padded-position `video_feat`/`audio_feat`/`text_feat`
   with large random noise (labels/mask unchanged) and re-ran forward.
   `logits` at all real positions were bit-for-bit unchanged
   (`allclose` True), and masked accuracy was unchanged (1.0 → 1.0). This is
   the structurally meaningful test: `SocialGNN.forward` only ever iterates
   `active = dialogue_mask[:, t].nonzero(...)`, so padded slots are never
   read into the recurrence at all — not just down-weighted, literally never
   touched.
3. ~~Label corruption at padded positions~~ — my first version of this
   check flipped padded labels from `-1` to `0` and observed the loss
   change (0.000003 → 0.277763), initially logged as "LEAK." **That's a
   flaw in the check, not a finding:** overwriting the `-1` sentinel removes
   the exact signal `FocalLoss`'s `ignore_index` filter uses to identify
   padding, so of course those positions get included afterward — the test
   was validating that the filter is *responsive to the sentinel value* (it
   is), not detecting a leak in the unmodified pipeline. The real,
   unmodified batch already has `-1` at every padded label (confirmed in
   check 1) and the un-corrupted loss (0.000003) reflects a clean fit over
   only the 58 real positions. No leak.

### Off-by-one / causality — PASS (direct gradient test, not just code reading)

Code reading (`SocialGNN.forward`) suggested no shift: `fused[:, t]` (built
from utterance t's own V/A/T) feeds the GRU update whose output becomes
`logits[:, t]`. Verified this directly and rigorously with a gradient
probe on a random 5-step dialogue: for every `t`, computed
`d(logits[0,t]) / d(video_input)` via autograd and checked which timesteps
have nonzero gradient.

```
logits[0,0]: depends_on_own_input=True  depends_on_FUTURE_input=False depends_on_past_input=False
logits[0,1]: depends_on_own_input=True  depends_on_FUTURE_input=False depends_on_past_input=True
logits[0,2]: depends_on_own_input=True  depends_on_FUTURE_input=False depends_on_past_input=True
logits[0,3]: depends_on_own_input=True  depends_on_FUTURE_input=False depends_on_past_input=True
logits[0,4]: depends_on_own_input=True  depends_on_FUTURE_input=False depends_on_past_input=True
```

Every prediction depends on its own timestep's input (never zero gradient
there) and legitimately on past timesteps (correct recurrence — earlier
context should influence later predictions), and **never** on future
timesteps. This is exactly the causal structure a correct implementation
should have, with no shift in either direction. Off-by-one ruled out with
direct evidence, not just code inspection.

### Graph-state isolation across batch elements — PASS

Two synthetic dialogues, both using `speaker_id=0` for every utterance
(deliberately colliding across batch elements) but with unrelated random
feature content (dialogue B scaled ×50 to make any cross-talk obvious).
Compared each dialogue's output computed alone vs. computed together in one
batch:

```
dialogue A (alone) matches dialogue A (batched with B): True
dialogue B (alone) matches dialogue B (batched with A): True
```

`speaker_hidden` in `SocialGNN.forward` is a fresh `list[dict]` created at
the top of every `forward()` call, one independent dict per batch index —
confirmed empirically that a shared `speaker_id` value across *different*
dialogues in the same batch does not cause any state leakage between them.

### Step 5 conclusion

**All five checks pass.** The model/collate/loss/masking/state-update code
is clean: no off-by-one, no padding leakage into loss or forward
computation, no cross-dialogue or cross-batch-element state leakage, and
the model can perfectly fit a small real sample given enough steps. This
rules out the entire "model/collate/masking/state-update" bug family from
Step 1's list. Combined with Step 4 ruling out CSV→index→cache-path
misalignment, **the training failure is fully attributable to the
feature-quality problems found in Steps 2–3** (random RoBERTa pooler,
audio padding leakage into the cached wav2vec2 features, and video
information loss from 8→16 frame duplication) — not to a bug in the
downstream model code.

**Next:** Step 6 synthesizes all findings, ranks root causes by evidence
strength, and proposes a minimal fix set — stopping there for sign-off
before implementing anything.

---

## STEP 6 — Synthesis

### What's ruled out

- **Model/collate/masking/state-update code** (Step 5): all 5 targeted
  checks pass — tiny-overfit reaches 100%, padding is excluded from both
  loss and forward computation, no off-by-one (direct causal gradient
  proof), no cross-dialogue/cross-batch-element state leakage.
- **CSV → processed-index → cache-key alignment** (Step 4): 10/10 clean on
  hand-picked unambiguous examples across 3 emotions.
- **Cache structural integrity** (Step 2): file counts match index rows
  exactly for every split/modality; no missing/duplicate files; the
  earlier collision fix (split-namespaced cache paths) holds.

None of the four candidate failure families from Step 1 — misalignment,
stale cache, off-by-one, padding leakage — turn out to be the actual bug.
**The real problem is feature quality, isolated to the frozen-backbone
wrapper / preprocessing layer, not anything downstream of the cache.**

### Root causes, ranked by evidence strength

**1. Text: RoBERTa `pooler_output` is randomly re-initialized every process (strongest evidence).**
Directly reproduced (two fresh backbone instances, identical input,
outputs differ by up to 0.93 — Step 2), directly explains the cache-vs-live
mismatch (30/30 samples, max diff 1.14 — Step 2), and independently
predicted by the linear probe falling below the task's 0.45 materiality
threshold (0.40 — Step 3). Affects **100%** of utterances uniformly. This
was flagged as a suspicion in the prior discrepancy report; it's now a
directly-reproduced, fully-understood bug.

**2. Audio: wav2vec2 conv feature-encoder sees raw padding before the attention mask excludes it (strong evidence, partial effect).**
Directly reproduced on the worst offender (a 0.21s clip, feature drift 2.02
between isolated and heavily-padded-batch computation — Step 2). Affects
**37%** of a random 30-utterance sample in the cache-vs-live check (mostly
small drift, a few large outliers, worse for short clips), consistent with
the linear probe landing below its healthy range (0.29 vs 0.38–0.45 —
Step 3).

**3. Video: 8→16 frame duplication likely destroys information the Kinetics-pretrained positional encoding expects (plausible, less directly isolated).**
The cache itself is provably *correct* (0/30 mismatches — Step 2's
strongest clean result), so this is a feature-quality problem, not a
caching bug. The linear probe is the *worst* of the three modalities (0.19
vs 0.33–0.40 healthy — Step 3) despite that clean cache. Frame duplication
(carried over from the prior preprocessing phase, driven by MViTv2's fixed
16-frame positional encoding) is the leading explanation per the prior
discrepancy report, but unlike #1 and #2 this diagnosis didn't run a direct
ablation (e.g. re-extracting native 16 frames and comparing probe scores)
to experimentally isolate frame-duplication specifically as *the*
mechanism, as opposed to e.g. a domain mismatch between Kinetics action
recognition and low-motion sitcom dialogue. Ranked third by evidence
strength, even though its probe score is the lowest in absolute terms.

### Minimal fix set (proposed, NOT applied)

| # | Fix | Where | Cache rebuild? | Re-preprocessing? | Touches the head? |
|---|---|---|---|---|---|
| 1 | Text: use `last_hidden_state[:, 0]` (raw pretrained CLS) or mean-pool `last_hidden_state`, instead of `pooler_output`, in `RobertaBackbone.forward` | `src/rapport/models/backbones.py` | **Yes** | No | No |
| 2 | Audio: compute wav2vec2 features per-utterance (batch size 1) in `build_feature_cache.py`'s audio step, eliminating cross-utterance padding entirely | `scripts/build_feature_cache.py` | **Yes** | No | No |
| 3 | Video: re-extract 16 native frames per clip instead of uniformly sampling 8 and repeat-interleaving | `scripts/preprocess_meld.py` (`NUM_FRAMES`, frame-sampling), then `MViTv2Backbone` no longer needs the repeat-interleave workaround | **Yes** | **Yes** (regenerate `data/meld/processed/frames/`) | No |

**All three fixes require a cache rebuild. None touch `SocialGNN`,
`collate_dialogues`, `FocalLoss`, or `trainer.py`** — Step 5 already showed
that code is correct, so there is nothing to fix there. Fix 3 is
categorically more expensive than 1 and 2: it requires re-running frame
extraction over all 13,706 clips (an earlier pipeline stage) before the
cache can even be rebuilt, whereas 1 and 2 only require re-running
`build_feature_cache.py` against the already-correct preprocessed
frames/wavs.

Suggested sequencing if/when signed off: fix 1 and 2 together (both are
cache-builder-only changes, cheap to combine into one rebuild), evaluate
impact via the same linear probes from Step 3 before deciding whether fix 3
is still worth its higher cost, or whether the resulting text/audio
improvement alone is enough to clear the hard floor and get a legitimate
`speaker_only` baseline number.

**STOPPING HERE for sign-off. No fixes have been applied — `backbones.py`,
`build_feature_cache.py`, `preprocess_meld.py`, and the cache on disk are
all unchanged from before this diagnosis started** (aside from the Step 2
rebuild, which reproduced byte-identical file counts and used the
already-fixed split-namespaced code — no behavior change).
