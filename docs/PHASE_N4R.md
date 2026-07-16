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
