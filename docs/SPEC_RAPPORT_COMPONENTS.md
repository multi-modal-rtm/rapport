# SPEC: RAPPORT components (relational memory, shift objective, temporal attention) — v1.0, adapted to the Phase T foundation

Saved verbatim as provided at the start of PHASE N4, before any
implementation. This is the authoritative mechanical spec for the
relational edge memory, emotion-shift auxiliary objective, and temporal
attention pooling components. Where it is silent on a detail, the
implementation makes a choice and records it in the "Implementation
decisions" appendix at the bottom of this file, rather than leaving the
choice undocumented.

**v1.1 addendum (Phase N4-R, `docs/PHASE_N4R.md`): see section D below.**
Sections A-C (relational memory, shift objective, temporal attention) are
UNCHANGED mechanically -- v1.1 only changes how the stack's output
combines with the frozen text classifier's own logits, and how the fusion
projector's audio/video blocks are initialized. Everything in A-C below
still describes the actual, current implementation.

## D. Residual "do-no-harm" redesign (v1.1)

Phase N4's full 15-run ablation matrix (`docs/PHASE_N4_STEP6.md`) found a
clean, monotonic HARD-gate failure: `base_fusion` (no relational/shift/
temporal) beat every config that added any of the three components, and
`full` (all three) was the second-worst of five. Phase N4-R's diagnosis
(`docs/PHASE_N4R.md` Steps 1-2) traced this primarily to overfitting --
train-vs-val gap grew monotonically with component count (mean +0.129 for
`base_fusion` -> +0.225 for `full`) -- with a smaller, secondary
contribution from the k=8 contextual text encoder already covering most of
what graph-level recency context would add.

**D1. Cached text logits.** The frozen Phase T checkpoint's own 7-d
classifier output, `z_text`, is cached per utterance alongside the 768-d
embedding (`scripts/build_text_ctx_cache.py`, sibling `{subdir}_logits`
cache dir, same frozen forward pass -- no extra model calls).

**D2. Residual connection.** The multimodal/relational stack's final
layer (`RapportModel.classifier`, playing the role of `W_out` in
`z = z_text + W_out(g_t)`) is ZERO-INITIALIZED (weight and bias), where
`g_t` is exactly the same readout the non-residual classifier already
consumed (`[h_{s,t} || mean_j e_{sj,t}]` for `relational=True`, `h_{s,t}`
otherwise -- spec A6). Because `W_out=0` at construction, `W_out(g_t) = 0`
for ANY `g_t`, regardless of how randomly-initialized the rest of the
stack is -- the emotion logits equal `z_text` exactly at t=0, for every
combination of relational/shift/temporal.

**D3. Fusion projector A/V blocks.** The fusion layer's audio and video
column blocks (`fusion[0].weight[:, 768:2304]`) are ALSO zero-init (text's
block keeps the default `nn.Linear` init); audio/video enter only as
learned corrections. This is not required for D2's zero-at-init guarantee
(which holds regardless, since `W_out=0` erases anything upstream) -- it's
a separate choice about early-training dynamics, so the model's first
gradient steps improve the text-only prediction rather than immediately
mixing in unlearned A/V noise.

**D4. Property test (the new invariant).** At initialization, every
config's predictions equal the Phase T model's predictions EXACTLY (not
approximately -- `torch.equal`), for every combination of relational/
shift/temporal. `tests/test_rapport_model_residual.py`, parametrized over
all 8 flag combinations.

**D5. Everything else unchanged from v1.0**: shift head (spec B, lambda
0.5, selection on val macro F1), relational memory (spec A), temporal
attention (spec C), plain CE loss, optimizer/schedule/dropout (RECIPE.md
locked GNN values).

### Implementation decisions (v1.1)

**Where the residual add happens.** `logits = logits + text_logits` is
applied in `RapportModel.forward`, OUTSIDE `_forward_base`/
`_forward_relational` -- neither of those two methods needs to know
`residual` exists at all. This preserves spec A7's bit-for-bit guarantee
for `_forward_base` without any extra conditional logic inside it, and
keeps the residual mechanism entirely orthogonal to which of the two node
paths is active.

**Fusion bias is NOT zeroed.** Only the audio/video weight COLUMNS are
zero-init (D3); the fusion layer's bias term isn't tied to any one
modality, so it keeps the default `nn.Linear` init.

**Shift head is untouched by the residual mechanism.** There's no
"Phase-T shift baseline" to residual against (shift is a genuinely new
signal), so `shift_head` keeps its ordinary (non-zero) init regardless of
`residual`.

---

## A. RELATIONAL EDGE MEMORY

**A1. State:** for a dialogue with N speakers, maintain edge states
`E_t ∈ R^{P x d_e}`, `P = N(N-1)/2` UNORDERED pairs, `d_e = 128` (locked).
Canonical helper `pair_index(i,j)` with i<j; unit-tested for bijectivity
and symmetry (`pair_index(i,j) == pair_index(j,i)`).

**A2. Lifecycle:** zero-init at every dialogue start. Edge states must
never leak across dialogues or across batch elements (same reset
discipline and test pattern as node states).

**A3. Update:** when speaker s utters at timestep t with utterance
embedding `U_{e,t}`, update ONLY edges incident to s. For each other
speaker j present in the dialogue:

```
e_{sj,t} = GRUCell([U_{e,t} || h_{s,t-1} || h_{j,t-1}], e_{sj,t-1})
```

ONE shared GRUCell across all pairs (weight sharing, no per-pair params).
Non-incident edges carry over unchanged.

**A4. Ordering (fixed, documented in the module docstring):** edge
updates read node states at t-1; the node update reads edge states at t.
Changing this ordering changes the model — it is locked.

**A5. Edge-conditioned message passing (GATv2-style):** attention logit
for j -> s incorporates the edge state:

```
alpha_{sj} ∝ a^T LeakyReLU(W_n h_j + W_n h_s + W_e e_{sj,t})
```

message: `m_{sj} = W_m [h_j || e_{sj,t}]`.

**A6. Readout:** classify from `[h_{s,t} || mean_j e_{sj,t}]` — the
speaker state concatenated with that speaker's aggregated relationship
context. This is what makes relationships first-class in the output path,
not only in message passing.

**A7. Equivalence guarantee:** with `relational=false`, the forward path
must reproduce the `base_fusion` path BIT-FOR-BIT on fixed input and
identical weights (unit test; protects all ablation deltas).

**A8. Dyadic degeneracy:** unit test N=2 (one edge) runs without
special-casing — IEMOCAP-readiness.

**A9. Tests (minimum):** pair indexing; per-dialogue reset; no
cross-batch leakage; incident-only updates (non-incident edges unchanged
after a step); ordering conformance; A7 equivalence; N=2; gradient flow
to all relational parameters.

## B. EMOTION-SHIFT AUXILIARY OBJECTIVE

**B1. Label derivation (preprocessing-time, into the index parquet —
never on the fly):** `shift_t = 1` iff utterance t's emotion label
differs from THE SAME SPEAKER'S previous utterance label within the
dialogue; 0 if same. A speaker's FIRST utterance in a dialogue has NO
shift label — masked from loss and metrics (consistent with the collate
padding mask). New parquet columns: `shift_label`, `shift_mask`.

**B2. Statistics:** produce `docs/SHIFT_LABEL_STATS.md` (shift rate
overall and per emotion class, per split). FLAG loudly if minority
classes (fear, disgust, surprise) do NOT show elevated shift rates vs
neutral — that is the method's premise and its failure changes the
paper's wording.

**B3. Model:** single-logit linear head on the current speaker's updated
state `h_{s,t}`; `BCEWithLogitsLoss` with `pos_weight` from the
train-split shift rate.

**B4. Total loss:** `L = L_CE_emotion + 0.5 * L_shift` (lambda = 0.5,
locked; plain CE per the Phase T recipe — no focal, no trained LA).

**B5. Selection:** early stopping and checkpoint selection remain on
EMOTION val macro F1 only. Shift F1 (binary) is logged every epoch as an
auxiliary metric, never a selection criterion.

**B6. Tests:** label derivation on a hand-built 3-speaker toy dialogue
covering — a speaker returning after others spoke; first utterances
masked; consecutive same-label utterances = 0; masked positions
contribute exactly zero gradient; `pos_weight` computation.

## C. TEMPORAL ATTENTION (audio + video streams only; text_ctx is exempt — its pooling is already contextual)

**C1. Module `TemporalAttentionPool`:** input token sequence `[B, L, 768]`
with padding mask; prepend a learnable aggregation token; ONE block of
multi-head self-attention (4 heads) + residual + LayerNorm; output = the
aggregation-token position.

**C2. Mean-pool-equivalent initialization (locked):** at step 0 the
module's output must approximate the plain mean over unpadded tokens.
Suggested scheme: zero-init the query projection (uniform attention) and
zero-init the attention block's output projection so the residual path
dominates — but ANY scheme passing C3 is acceptable.

**C3. Tests:** at init, output within atol 1e-5 (fp32) of the masked mean
for random inputs; padded positions do not influence the output; gradient
flows to all parameters after one backward.

**C4. Consumes the cached A/V token sequences (frozen backbones);
parameter count reported in the trainable-params breakdown.**

---

## Implementation decisions

Filled in during implementation, whenever the spec above is silent on a
concrete choice. Each entry: what was undecided, what was chosen, why.

### A. Relational edge memory: how the attention context feeds the node GRU

**Undecided:** section A5 specifies the GATv2-style attention/message
formulas and says (A4) "the node update reads edge states t", but doesn't
pin down exactly how the resulting attention context combines with the
speaker's own previous hidden state inside the node GRU update.

**Chosen:** the edge-conditioned attention context `context_s` (aggregated
from OTHER speakers only -- edges are never self-pairs, so there's no
self-loop) is concatenated with the utterance embedding `U_e` and fed as
the GRUCell's **input** (`nn.GRUCell(FUSION_DIM + EDGE_DIM, HIDDEN_DIM)`),
while the speaker's own raw previous state `h_{s,t-1}` remains the GRUCell's
**hidden** argument, unchanged in kind from the base_fusion path. Contrast
with the non-relational path, where the plain (self-attending)
`GraphAttentionLayer` conflates "my own history" and "everyone's history,
attention-weighted" into a single value that becomes the GRU's hidden
input.

**Why:** relational edges by construction carry no self-information (A1: P
= N(N-1)/2, no i=i pairs), so there is no edge-based way to represent "my
own history" the way the base GAT's self-attention incidentally can. Since
the node update must still remember its own trajectory, the cleanest
separation of concerns is: hidden state = pure recurrent self-memory
(matches the non-relational path's role for the GRU's hidden argument),
input = "what I said this turn" + "what I'm picking up about my
relationships" (the new edge-conditioned signal). This keeps the two
information sources distinguishable rather than pre-mixed, and requires no
change to how h_{s,t-1} is retrieved/stored (same dict-based per-speaker
state as the base path) -- only the GRUCell's input size and the source of
its second input tensor differ. Implemented in
`rapport.models.rapport_model.RapportModel._forward_relational`.

### A6. Classifier input dimension

**Undecided:** the spec's readout formula, `[h_{s,t} || mean_j e_{sj,t}]`,
implies a classifier input dimension of `HIDDEN_DIM + EDGE_DIM` for the
relational path, larger than the non-relational path's `HIDDEN_DIM` alone,
but doesn't say whether to keep a separate classifier per path or project
down to a shared dimension first.

**Chosen:** a separate `nn.Linear` classifier per path (`HIDDEN_DIM` in
for `relational=False`, `HIDDEN_DIM + EDGE_DIM` in for `relational=True`),
not a shared classifier with a projection layer. This is the simplest
option that satisfies A7 (relational=False's classifier must be bit-for-bit
identical to Step 1's, which only ever saw a `HIDDEN_DIM`-wide input) with
zero risk of an accidental shape/projection interaction between the two
paths.

### B. Shift head's return convention

**Undecided:** the spec doesn't say how the shift logit should be exposed
from the model's forward pass alongside the emotion logits.

**Chosen:** `RapportModel.forward` now always returns a 2-tuple
`(emotion_logits, shift_logits)`, where `shift_logits` is `None` whenever
`shift=False`. This is a one-time, additive change to the return
convention (made once, at the point `shift` was implemented, rather than
threading an optional third return value through every future step) --
every caller (tests, `scripts/train_rapport.py`) unpacks the tuple
regardless of whether `shift` is enabled for that particular model
instance, so there's exactly one calling convention to remember.

### B3. Shift head input across the relational/non-relational split

**Undecided:** whether the shift head should see the same
possibly-edge-augmented state the emotion classifier sees, or something
narrower.

**Chosen:** exactly as spec B3 states literally -- the shift head is
`nn.Linear(HIDDEN_DIM, 1)` applied to the RAW node state `h_{s,t}`
(`new_hidden_batch` in the code) in BOTH the relational and non-relational
paths, never the edge-augmented `[h_{s,t} || mean_j e_{sj,t}]` the emotion
classifier sees when `relational=True`. This keeps the shift head's input
dimension and semantics identical regardless of the `relational` flag,
consistent with spec B3's specific wording ("the current speaker's updated
state h_{s,t}", not the readout vector spec A6 defines separately for the
emotion classifier).

### C2. Mean-pool-equivalent init scheme

**Undecided (explicitly, per spec: "ANY scheme passing C3 is acceptable"):**
how to make a "residual + LayerNorm" attention block behave as a masked
mean at init.

**Chosen, and why the obvious approach doesn't work:** a literal
post-residual LayerNorm (`LayerNorm(agg + MHA(agg, tokens))`) can NEVER
exactly reproduce the raw masked mean for arbitrary inputs, at any point
in training, because LayerNorm's normalization step (subtract the
per-sample mean across features, divide by the per-sample std) has no
learnable parameter that can turn it into an identity map -- only the
post-normalization affine (default weight=1, bias=0) is learnable, and
that alone can't undo the normalization itself. So instead: LayerNorm is
applied to the QUERY/KEY path only (`qk_norm`, used to compute attention
scores), never to the VALUE path or the final residual sum -- a "QK-norm"
variant, a real, previously-used transformer design, not an invented
workaround. Combined with zero-initializing the query projection (->
attention scores are exactly 0 for every unmasked position regardless of
what LayerNorm did to the keys, so softmax gives exactly uniform weight
over unmasked tokens) and identity-initializing the value/output
projections, the block's output is EXACTLY `agg_token + masked_mean(tokens)`
at init, and `agg_token` is itself zero-init, giving exact (not
approximate) masked-mean equivalence -- verified to atol 1e-5 for both
padded and unpadded random inputs (`tests/test_temporal_attention.py`).

One consequence, noted rather than hidden: because the query path is
multiplied by an all-zero weight matrix at init, the gradient flowing back
into `qk_norm`'s parameters on the very first backward pass is exactly
zero (not None -- PyTorch still populates a defined, all-zero gradient
tensor, which is what `tests/test_temporal_attention.py`'s
`test_gradient_flows_to_all_parameters` checks for, matching this
project's convention elsewhere, e.g. `tests/test_relational_memory.py`).
This is the standard, well-understood behavior of any "zero-init" trick
(e.g. the same GRUCell/GAT parameter-sharing tricks used elsewhere in
Phase N4) -- the pathway "wakes up" after the first optimizer step moves
`q_proj` away from exactly zero, not a bug.

### C4. One temporal pool per modality, not shared

**Undecided:** whether audio and video should share one
`TemporalAttentionPool` instance or each get their own.

**Chosen:** separate instances (`RapportModel.video_temporal_pool`,
`.audio_temporal_pool`), not shared weights. The spec doesn't say to
share, and audio (wav2vec2) and video (MViTv2) token sequences come from
different frozen backbones with different statistics -- sharing a pooling
module's weights across them would impose an assumption (that the same
learned attention pattern transfers across modalities) the spec never
asked for. The extra parameter cost is small (two `TemporalAttentionPool`
instances vs. one, each dominated by four 768x768 linear layers).

**Also implemented:** batch-padding dialogue positions (a shorter
dialogue's padding, when batched alongside a longer one) have an
all-False token mask (no real tokens were ever written there) -- feeding
that directly into the pool's softmax would divide 0/0 (NaN), which would
then contaminate every OTHER row's gradient for the pool's shared
parameters once summed during backward. `RapportModel._temporal_pool_av`
filters to only the utterance positions `dialogue_mask` marks real before
pooling, then scatters the result back into an all-zero `[B, L, 768]`
tensor via `index_copy` (differentiable, unlike in-place indexed
assignment) -- covered by
`tests/test_rapport_model_temporal.py::test_temporal_handles_batch_padding_without_nan`.
