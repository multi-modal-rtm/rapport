# PHASE N4-R — diagnosis, then residual "do-no-harm" redesign

Follow-up to Phase N4's clean HARD-gate failure (`docs/PHASE_N4_STEP6.md`):
`base_fusion` beat every config with relational/shift/temporal components,
including `full`. Two candidate causes: (1) the new components had to
re-learn the text foundation through randomly-initialized parameters
instead of starting from it; (2) the k=8 contextual text encoder already
captures most of what graph-level recency context would add, making the
GNN's marginal contribution small or negative. Tested both before
redesigning.

---

## STEP 1 — Diagnostics from existing Phase N4 artifacts (no training)

`scripts/diagnose_n4_gate_failure.py`: forward-pass-only evaluation of the
15 existing checkpoints (no gradient updates) plus direct analysis of
already-logged history/weights. Full numbers:
`outputs/n4r_step1_diagnostics.json`.

### 1a. Train-vs-val gap per config/seed — clean, monotonic overfitting fingerprint

Each checkpoint's TRAIN-split weighted F1 (forward pass only) minus its
own selected-checkpoint VAL weighted F1:

| config | n_components | seed 42 gap | seed 1337 gap | seed 2024 gap | **mean gap** |
|---|---|---|---|---|---|
| base_fusion | 0 | +0.1218 | +0.1407 | +0.1246 | **+0.1290** |
| minus_temporal | 2 (rel+shift) | +0.2309 | +0.1795 | +0.1337 | **+0.1813** |
| minus_shift | 2 (rel+temporal) | +0.2229 | +0.2059 | +0.1649 | **+0.1979** |
| minus_relational | 2 (shift+temporal) | +0.2821 | +0.2319 | +0.0892 | **+0.2011** |
| full | 3 | +0.3023 | +0.1864 | +0.1872 | **+0.2253** |

**The mean gap increases monotonically with component count** (0.129 ->
~0.18-0.20 for any 2-component config -> 0.225 for `full`), with the
single largest individual gap (seed 42, `full`: +0.3023, train weighted
F1 0.879 vs val 0.577) belonging to the 3-component config. This is
exactly the overfitting fingerprint cause (1) predicts: every added
component is extra randomly-initialized capacity sitting on top of a
frozen text encoder, and that capacity is being used to memorize the
training set rather than generalize. `base_fusion` already overfits
somewhat (+0.129 is not small for a model this size on ~10K training
utterances), but every additional component makes it measurably worse.

### 1b. Learning curves: did `full` ever exceed base_fusion's peak val metric?

| seed | full peak val weighted F1 | base_fusion peak val weighted F1 | ever exceeded (weighted)? | full peak val macro F1 | base_fusion peak val macro F1 | ever exceeded (macro)? |
|---|---|---|---|---|---|---|
| 42 | 0.6001 | 0.6053 | **No** | 0.4225 | 0.4607 | **No** |
| 1337 | 0.5975 | 0.6021 | **No** | 0.4345 | 0.4696 | **No** |
| 2024 | 0.6082 | 0.6071 | Yes (+0.0011, marginal) | 0.4370 | 0.4674 | **No** |

**`full` never exceeded `base_fusion` on val macro F1 at any epoch, in any
seed** — not a single crossing point across an entire training run. On
weighted F1 it crossed marginally in exactly one seed (2024, by 0.0011,
noise-level). This is "dominated throughout," not "started behind and
caught up partway" — full curves in `outputs/n4r_step1_diagnostics.json`.

### 1c. Did the minority-classes-shift-more premise hold? (restated from `docs/SHIFT_LABEL_STATS.md`)

**Yes, clearly, confirmed in all 3 splits** — this was not the failure
point. Eligible-row shift rate, neutral vs. the three flagged minority
classes:

| split | neutral | fear | disgust | surprise |
|---|---|---|---|---|
| train | 0.3956 | 0.8413 | 0.8058 | 0.7503 |
| dev | 0.4304 | 0.6875 | 0.8000 | 0.7184 |
| test | 0.3877 | 0.9189 | 0.8478 | 0.7610 |

Fear/disgust/surprise all sit at 1.5-2.4x neutral's shift rate in every
split — the auxiliary label itself carries the intended signal. Step 1a/1b
point at *learning dynamics* (overfitting, never catching up to
`base_fusion`), not at the shift label's premise being wrong.

### 1d. base_fusion's learned fusion weights — no evidence of learned modality suppression

Frobenius norm of `W_proj`'s three 768-column blocks (text_ctx / audio /
video) in the first fusion linear layer, `base_fusion` checkpoints:

| seed | text_ctx norm (%) | audio norm (%) | video norm (%) |
|---|---|---|---|
| 42 | 6.076 (34.4%) | 5.704 (32.3%) | 5.879 (33.3%) |
| 1337 | 6.247 (34.2%) | 5.846 (32.0%) | 6.152 (33.7%) |
| 2024 | 6.183 (34.3%) | 5.799 (32.2%) | 6.041 (33.5%) |

**Essentially an even 3-way split, not a learned preference for
text_ctx** — despite text being by far the strongest single-modality
signal throughout this entire project (`docs/DIAGNOSIS.md`'s linear
probes: text 0.53 >> audio 0.41 > video 0.33 unbalanced weighted F1). This
doesn't mean the model ignores modality quality (weight *norm* in the
first linear layer is a weak proxy for a block's actual downstream
influence, which also depends on the GRU/GAT/classifier weights
downstream of it) — but it does rule out one simple, hoped-for
possibility: the fusion layer does not appear to have discovered
"down-weight the weaker modalities" on its own, at least not through this
one layer's weight magnitude.

### Step 1 conclusion: cause (1) has strong direct evidence; cause (2) untested until Step 2

The overfitting-gap result (1a) and the "never caught up" learning curves
(1b) are both squarely consistent with cause (1) -- re-learning/adapting
around the frozen text foundation through random-init capacity is costing
generalization, and the cost scales with how much such capacity is added.
Nothing in Step 1 speaks to cause (2) (contextual-encoding redundancy)
directly -- that requires the k=0 comparison in Step 2.

---

## STEP 2 — Redundancy probe: k=0 text cache, one run

`scripts/build_text_ctx_cache.py --k 0 --out-subdir text_ctx_k0`: the SAME
frozen Phase T encoder (identical checkpoint, identical weights), fed
context-free (k=0, current utterance alone) input instead of the default
k=8, written to a separate cache subdir. `scripts/train_rapport.py`
gained a `--text_cache_subdir` override so `base_fusion` could be trained
against this k=0 cache without touching the k=8 cache every other config
depends on. One run: `base_fusion` on the k=0 cache, seed 42 only.

### Anchor definitions (apples-to-apples with the already-documented k=8 gain)

The k=8 "gain" already on record (`docs/PHASE_N4_STEP6.md`'s
component-attribution table) is `base_fusion`'s own weighted F1 minus the
Phase T model's OWN trained performance -- not a linear-probe baseline.
For a fair comparison, the k=0 anchor uses the same definition: the
EXISTING Phase T seed-42 model (`context_text_plain_ce_seed42`, trained at
k=8), evaluated with its own already-trained classifier head on
context-free (k=0) test inputs -- zero additional training, just a
different input at inference time.

| | k=8 | k=0 |
|---|---|---|
| text-only anchor (Phase T model's own weighted F1, seed 42) | 0.6353 | 0.6234 (same model, fed k=0 inputs) |
| `base_fusion` weighted F1 (seed 42) | 0.6345 | 0.6381 |
| **GNN gain (base_fusion − anchor)** | **-0.0008** | **+0.0147** |

(The k=0 anchor, 0.6234, is itself lower than the k=8 anchor, 0.6353 --
expected: the Phase T model was trained to expect and use k=8 context, so
depriving it of context at inference degrades its own accuracy by ~0.012,
a sanity-check result in the right direction.)

### Interpretation — directionally consistent with cause (2), but the effect is modest, not dramatic

**The GNN's gain over the text-only anchor is larger at k=0 (+0.0147)
than at k=8 (-0.0008)** — an ~0.0155 swing in the GNN's favor when the
text features it receives carry no context of their own. This is
directionally exactly what cause (2) predicts (contextual encoding
substitutes for graph-level recency context, so removing the former
should increase the latter's marginal value) and is being recorded
honestly as **a real, positive, but modest effect, not the "large gain"
that would make cause (2) the dominant or sole explanation.** +0.0147 is
about 1.5 weighted-F1 points on a single seed -- a genuine signal, not
noise-level (Phase N4's inter-seed std for a fixed config was typically
0.005-0.015), but nowhere near large enough on its own to explain Phase
N4's full ablation-matrix failure (where `full`'s mean gap below
`base_fusion` was -0.0129, an order of magnitude larger than this
+0.0147 k=0 recovery).

**Conclusion: both causes have real evidence; cause (1) (overfitting from
random-init capacity, Step 1) is the stronger, better-supported
explanation for the magnitude of Phase N4's failure, and cause (2)
(contextual-encoding redundancy) is a real, secondary, worth-recording
effect** -- consistent with the pre-registered framing that relational
memory specifically "must justify itself on LONG-RANGE relationship
signal" going forward, since its SHORT-RANGE recency contribution does
appear to be substantially covered by the k=8 contextual text encoder
already. Both findings motivate Step 3's redesign: making random-init
capacity provably harmless at initialization (directly addressing cause
1) rather than attempting to further re-tune context window sizes
(which would only partially address cause 2 and wasn't the
larger-magnitude problem anyway).

---

## STEP 3 — Residual "do-no-harm" redesign (spec v1.1)

Full mechanical spec: `docs/SPEC_RAPPORT_COMPONENTS.md` section D.
Summary: `z = z_text + W_out(g_t)`, `W_out` (the model's final
classification layer) zero-init, fusion projector's audio/video column
blocks also zero-init. This directly targets cause (1) -- the stack can no
longer regress below the frozen text baseline at initialization, by
construction, regardless of how much random-init capacity is added.

**Implementation:**
- `scripts/build_text_ctx_cache.py` now also caches `z_text` (the frozen
  Phase T classifier's own 7-d logits) alongside the 768-d embedding, to a
  sibling `{subdir}_logits` cache dir -- rerun for the k=8 cache
  (`text_ctx_logits/` now exists alongside `text_ctx/`).
- `RapportModel(residual=True)`: zero-inits `classifier` (weight + bias)
  and `fusion[0].weight`'s audio/video column blocks; `forward()` gained
  an optional `text_logits` argument, added to the stack's own logits
  OUTSIDE `_forward_base`/`_forward_relational` (so spec A7's bit-for-bit
  guarantee needs no changes).
- `MELDCachedDataset(load_text_logits=True)` + `collate_dialogues` thread
  `text_logits` through the data pipeline; `scripts/train_rapport.py`
  gained `--residual` and passes `text_logits` into the model call when
  set.

**Property test** (`tests/test_rapport_model_residual.py`, 12 tests):
parametrized over all 8 relational/shift/temporal combinations, asserts
`torch.equal(logits, text_logits)` at initialization with RANDOM (not
zeroed) A/V input -- confirming the guarantee holds regardless of what the
rest of the stack does, not just in the specific all-zero-input case.
Also verifies: the correction becomes nonzero after a few optimizer steps
(the property is init-only, not a permanent no-op); gradients reach every
parameter including the relational/shift heads; the fusion A/V blocks are
zero-init when `residual=True` and NOT zero-init when `residual=False`
(regression guard against the flag silently doing nothing).

**Smoke-test confirmation the design works as intended:** a 2-epoch
`full_R` (relational+shift+temporal+residual) run reached val macro F1
0.44-0.45 by epoch 0-1 -- compare to the original (non-residual) `full`
config, which started near 0.30-0.33 at the same point
(`docs/PHASE_N4.md` Step 5 logs). Starting from the text classifier's own
already-decent performance instead of from scratch is exactly the
intended effect.

**Step 3 complete.** Next: Step 4, the compact 9-run re-evaluation.

---

## STEP 4 — Compact re-run: gates + decision

`scripts/run_ablation_matrix.py --configs base_fusion_R full_R
minus_relational_R` (9 runs), `scripts/report_n4r_step4.py` for the gate
check. Full table: `docs/PHASE_N4R_STEP4.md`.

### Gate (a) — HARD, structural: no config below 0.6358

| config | mean raw weighted F1 | std | pass? |
|---|---|---|---|
| base_fusion_R | 0.6392 | 0.0011 | **PASS** |
| minus_relational_R | 0.6365 | 0.0032 | **PASS** |
| full_R | 0.6344 | 0.0061 | **FAIL** (by 0.0014) |

**Gate (a) overall: FAIL** -- one of three configs (`full_R`) misses the
floor, narrowly (0.0014, well inside a single seed's typical std of
0.005-0.01 for this recipe).

### Gate (b) — DECISION: full_R vs base_fusion_R

| metric | delta (full_R − base_fusion_R) |
|---|---|
| raw weighted F1 | **-0.0048** |
| raw macro F1 | -0.0033 |
| fear F1 | +0.0056 (full_R 0.079 vs. base_fusion_R 0.074) |
| disgust F1 | -0.0186 (full_R 0.179 vs. base_fusion_R 0.198) |

### Synthesis — dramatic improvement, honestly still not a clean pass

**The residual redesign worked exactly as intended on its primary
target.** The full-vs-base_fusion gap shrank by ~63%: from **-0.0129**
(Phase N4's original architecture, `docs/PHASE_N4_STEP6.md`) to
**-0.0048** (`full_R` vs. `base_fusion_R` here) -- and the absolute
weighted F1 numbers moved from a 0.60-0.63 range clustered well below the
text-only anchor, to a tight 0.63-0.64 cluster essentially AT the anchor
(deltas of -0.001 to -0.006, vs. the original matrix's -0.005 to -0.022).
This is strong, direct confirmation that cause (1) (random-init capacity
overfitting) was the dominant driver of Phase N4's failure -- making that
capacity provably harmless at initialization fixed most of the damage.

**But per the pre-registered reading of gate (b): this is still an
analyzed negative result for relational memory (+shift, given
`full_R` bundles both against `base_fusion_R`), not a demonstrated
contribution.** `full_R` still underperforms `base_fusion_R` on the
primary metric (weighted F1, -0.0048) and on macro F1 (-0.0033); the
per-class picture is mixed rather than a clear win -- fear improves
marginally (+0.0056) but disgust gets measurably worse (-0.0186), and
`minus_relational_R` (shift+temporal, no relational) actually beats
`full_R` on every metric checked (weighted F1, fear F1, disgust F1),
suggesting relational memory's specific marginal contribution, once
overfitting is controlled for, is at best neutral and possibly still
mildly negative -- consistent with Phase N4-R Step 2's finding that its
short-range recency signal is largely redundant with the k=8 contextual
text encoder, and that any real justification for it would need to come
from long-range relationship signal the current recipe/dataset/eval
doesn't clearly surface.

**Gate (a) technically fails** (one config, `full_R`, misses the
structural floor by 0.0014) -- reported exactly as it is, not rounded up
to a pass. Per instructions, this is the row that decides the paper
framing: **relational memory and the shift objective, as currently
specified, are reported as an analyzed negative result** (a real,
substantially-improved-but-still-negative delta, with a clear, evidenced
mechanism -- overfitting was the dominant cause and is now controlled
for, but the components still don't earn their keep even once that
control is in place) rather than as a positive contribution.

**Stopping here per instructions.** No further tuning or additional runs
follow from this result without explicit direction.
