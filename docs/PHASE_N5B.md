# PHASE N5-B — IEMOCAP: the relational hypothesis's fair trial

**STATUS: UNBLOCKED, text anchor built and GATE PASSED** (Steps B1-B4
complete as of this doc; relational-memory matrix -- Step B5, the actual
test of the pre-registered hypothesis below -- not yet run).

## Original blocker (historical record, resolved)

Originally blocked: no IEMOCAP data, no download script, no credentials on
this machine -- only a declarative stub (`configs/data/iemocap.yaml`).
Unblocked when the release became available at
`~/data/iemocap/iemocap/IEMOCAP_full_release` (md5
`521be1e5eec425ae21fdc27c763ca813`, verified before extraction).

## Pre-registered hypothesis (quoted verbatim, predates any IEMOCAP result)

> Pre-registered hypothesis (stated here, before any IEMOCAP result
> exists, so it can't be fitted after the fact): **relational memory's
> delta on IEMOCAP > its delta on MELD, driven by dialogue length**
> (IEMOCAP dialogues run substantially longer than MELD's, which should
> give graph-level relationship state more room to matter than MELD's
> `docs/PHASE_N4R.md`/`docs/PHASE_N5A.md` results showed). Either outcome
> -- confirmed or not -- gets reported when this phase actually runs.
> Also log the single dyadic edge-state norm over time for 3 sample
> sessions as a qualitative figure candidate, once relational memory is
> running on real IEMOCAP dialogues.

**Not yet tested** -- this doc covers B1-B4 (inventory through text
anchor) only. The residual matrix that actually tests this hypothesis
(`base_fusion_R`/`full_R`/`minus_relational_R` x 3 seeds) is Step B5,
not run in this pass.

## Step B1 — corpus inventory

Full detail: `docs/iemocap_inventory.md` (`scripts/iemocap_inventory.py`).
Corpus counts matched published expectations exactly: 151 dialogues,
10,039 utterances, 10 (session,gender) speaker slots, every dialogue has
all four file types (wav/avi/transcription/EmoEvaluation).

## Step B2 — label protocol decision

**Decided: 6-class** {angry, happy, excited, sad, neutral, frustrated}.
Full rationale and the counts-table evidence: `docs/RECIPE.md`'s "IEMOCAP
-- label protocol decision" section. Split: Session5=test, Session4=val,
Sessions1-3=train.

## Step B3 — preprocessing

`scripts/preprocess_iemocap.py` mirrors `scripts/preprocess_meld.py`'s
output schema exactly (`dialogue_id, utterance_id, speaker, speaker_id,
emotion, label, text, frame_path, wav_path, shift_label, shift_mask`) so
every existing `MELDCachedDataset`/`MELDRawDataset`/`build_feature_cache.py`
consumer works against it unmodified.

**What's different from MELD, and why:** IEMOCAP's release is
DIALOGUE-level (one wav + one avi per dialogue), not pre-cut per-utterance
clips like MELD.Raw -- this script actually cuts each utterance's audio
(ffmpeg, using the EmoEvaluation `[start,end]` timestamps) and video (8
frames uniformly sampled within `[start,end]`, decoded in a single
streaming PyAV pass per dialogue keyed to only the frame indices any
utterance in that dialogue needs, to bound memory instead of decoding
every frame of a multi-minute dialogue video).

**Split-screen speaker cropping:** IEMOCAP's video is a side-by-side
split screen of both actors. Verified visually (4 sample frames across
sessions 1/3/5, both F- and M-designated dialogue files -- see the
extraction commands in this phase's history) that the actor wearing MoCap
markers -- i.e. whichever gender letter appears in the dialogue's own name
(`Ses01F_impro01` -> F) -- is **always on the LEFT half**, the other actor
always on the right, consistently across every session checked. Each
utterance's frames are cropped to whichever half the SPEAKING actor is on
(compares the utterance's own speaker gender, parsed from its turn id,
against the dialogue's designation letter -- not the dialogue's letter
alone, since both actors speak within one dialogue file).

**Result:** 7,380 utterances kept (6-class filter applied during
preprocessing, not after -- xxx/oth/fear/disgust/surprise utterances are
never cut/cached at all), split train=4,246 / val=1,512 / test=1,622 --
matching `docs/iemocap_inventory.md`'s independent audit exactly. **0 bad
dialogues** (every dialogue's wav/avi/transcription/EmoEvaluation
decoded/cut cleanly). `tests/test_iemocap_data.py` (7 tests, mirroring
`tests/test_meld_data.py`) all pass against the real processed index:
split counts match the inventory audit, all 6 labels present in every
split, dialogue ordering preserved, every dialogue confirmed dyadic (2
speakers), global speaker vocab has exactly 10 entries, frame/audio tensor
shapes and 16kHz-mono confirmed, collate padding/masking correct.

## Feature caches

`scripts/build_feature_cache.py --processed-dir data/iemocap/processed
--out-dir data/iemocap/cache --splits train val test` -- reused
UNMODIFIED (already fully parameterized via `--processed-dir`/`--out-dir`/
`--splits`, schema-generic). Built in under 3 minutes total (train 84.8s,
val 30.0s, test 32.0s).

**Process-independence check** (`scripts/audit_cache.py`, extended with a
`--splits` flag so it works for IEMOCAP's train/val/test naming instead of
MELD's train/dev/test -- diagnostics-only script, not a locked recipe, so
this minimal parameterization was safe to add without touching MELD's own
invocations): 20 random utterances, features recomputed live through fresh
backbone instances in a separate process from the one that built the
cache, compared against the on-disk cached tensors. **video:
max_abs_diff=0.000000, audio: max_abs_diff=0.000000, text:
max_abs_diff=0.000004 -- all `allclose(atol=1e-3, rtol=1e-3)=True`**,
matching MELD's own established process-independence result
(`docs/DIAGNOSIS.md`) almost exactly.

(The same script's cross-split key-overlap check reports collisions
between train/val/test -- this is expected and harmless, not a new issue:
`dialogue_id` restarts at 0 per split by design (mirroring
`preprocess_meld.py`'s own convention), so the bare `dia{X}_utt{Y}` stem
is not globally unique across splits -- the real uniqueness guarantee is
the split-named subdirectory. Verified this is pre-existing MELD behavior
too, not something introduced here, by running the same check against
`data/meld/cache` directly.)

## Step B4 — the IEMOCAP text anchor

Identical recipe to Phase T (`docs/RECIPE.md`'s frozen Phase T recipe,
UNCHANGED): roberta-base + LoRA (r=8, alpha=16, dropout=0.05,
target=[query,value]), masked-mean pooling, plain CE, AdamW lr=2e-4,
linear warmup 10%, max 10 epochs, early stop patience 3 on val macro F1,
context k=8, max_length=256. `scripts/train_context_text_iemocap.py` is a
parallel script (NOT an edit to the frozen `scripts/train_context_text.py`)
reusing every shared component unchanged, repointed at
`data/iemocap/processed` and the 6-class label set.

**Post-hoc tau selection:** the locked selection rule (highest val macro
F1 subject to all-classes-nonzero AND val weighted F1 >=
`VAL_CANDIDATE_WEIGHTED_F1_FLOOR`) was applied with MELD's calibrated
floor (0.58), since no IEMOCAP-specific floor has been established yet.
**No candidate qualified on seed 42's val sweep** (best raw val weighted
F1 across all 5 swept taus topped out at 0.522, tau=0.25 -- all 5 had
all-classes-nonzero, none cleared 0.58). This is the correct, honest
output of the locked RULE applied to a differently-distributed val split,
not a bug: the 0.58 floor was calibrated specifically for MELD's 7-class
val behavior. Per this project's tau-freezing convention (`docs/RECIPE.md`
Amendment 2 -- select once, apply identically elsewhere), this "no
adjustment" decision was frozen for seeds 1337/2024 too
(`--freeze_no_posthoc`, added to the script rather than silently
re-running per-seed auto-selection, which would have violated the
freezing convention just as much as re-selecting a nonzero tau would).
**Every seed's headline numbers below are therefore RAW (unadjusted)
test metrics.**

### 3-seed table

| seed | test weighted F1 | test macro F1 | test accuracy | all 6 classes nonzero | best epoch |
|---|---|---|---|---|---|
| 42 | 0.5815 | 0.5729 | 0.5783 | True | 3 |
| 1337 | 0.5939 | 0.5738 | 0.5962 | True | 6 |
| 2024 | 0.5618 | 0.5480 | 0.5592 | True | 2 |
| **mean ± std** | **0.5790 ± 0.0162** | 0.5649 | 0.5779 | 3/3 | |

Per-class test F1 (seed 42): angry 0.518, happy 0.459, excited 0.546, sad
0.771, neutral 0.592, frustrated 0.551 -- no collapsed class; `sad` is the
strongest (largest single-class F1 across all 3 seeds), `happy` the
weakest (smallest kept class, 595 utterances corpus-wide), consistent with
the class-imbalance pattern already documented in
`docs/iemocap_inventory.md`.

### GATE: PASSED

| criterion | required | actual | pass? |
|---|---|---|---|
| anchor stable | std(test weighted F1) < 0.02 | 0.0162 | **yes** |
| all-6-nonzero | in >= 2/3 seeds | 3/3 | **yes** |

### Sanity context: where this anchor lands vs. published 6-class IEMOCAP text-only results

Published 6-class IEMOCAP text-only unimodal baselines commonly report
weighted F1 / accuracy in roughly the mid-50s to mid-60s percent range,
depending on model and protocol -- this anchor's 0.579 ± 0.016 weighted F1
sits within that broad neighborhood, a plausible result rather than an
outlier in either direction. **This is a sanity check, not a benchmark
claim**, for one explicit reason: **the split-convention caveat.** The
literature's dominant IEMOCAP protocol is **leave-one-session-out 5-fold
cross-validation** (average performance across 5 train/test splits, each
holding out one session), which reuses nearly all the data for both
training and (across folds) testing. This project's split is a **single,
fixed** Session5=test/Session4=val/Sessions1-3=train partition, stated in
advance and never tuned -- less data-efficient than 5-fold CV (only 3 of
5 sessions ever seen in training) and subject to whatever
Session5-specific quirks (speaker pair, session-level tone) a single held-
out fold carries that cross-validation would average away. The two
numbers are not directly comparable on a leaderboard basis; this anchor's
job is to establish that IEMOCAP text alone is TRAINABLE and STABLE under
this project's own fixed-split convention (the gate above), not to claim
a literature-competitive score under a different protocol.

## What's needed for Step B5 (not run in this pass)

1. A/V caches: **done** (see above).
2. Text anchor: **done** (see above), frozen checkpoint =
   `outputs/iemocap_text_anchor_seed42/best_model.pt` (matching the
   MELD Phase T convention of freezing the seed-42 run as the project's
   text encoder for downstream fusion configs).
3. Cache this text anchor's own pooled embeddings + classifier logits
   (mirroring `scripts/build_text_ctx_cache.py`'s role for Phase T) --
   not yet built.
4. The residual matrix `{base_fusion_R, full_R, minus_relational_R} x 3
   seeds`, then the pre-registered hypothesis comparison above.
5. The single dyadic edge-state norm over time for 3 sample sessions
   (qualitative figure candidate), once relational memory is running on
   real IEMOCAP dialogues.

**Nothing on the pre-registered hypothesis itself has run yet** -- B5 is
future work, not covered by this doc's GATE PASSED status (which applies
only to the text anchor's own stability, a precondition for B5, not the
hypothesis test itself).
