# PHASE N4 — RAPPORT core: multimodal fusion + relational memory + shift objective

Built on the certified Phase T text foundation (3-seed anchor 0.6403 ±
0.0045 raw weighted F1, `docs/PHASE_T_STEP4.md`). Two-stage design: the
text encoder is FROZEN at the Phase T checkpoint for the whole ablation
matrix; joint fine-tuning happens once, later, on the winning config only.
Mechanical spec for the relational/shift/temporal components:
`docs/SPEC_RAPPORT_COMPONENTS.md`.

---

## STEP 0 — Contextual text cache

### Frozen text encoder

**THE project text encoder for Phase N4 is the Phase T seed-42 checkpoint
under the frozen plain-CE + post-hoc recipe** (`docs/RECIPE.md`'s Phase T
section) — not the earlier, superseded LA-trained checkpoint.

| | |
|---|---|
| checkpoint | `outputs/context_text_plain_ce_seed42/best_model.pt` |
| sha256 | `447f369f02aad5297e7050a41f0ac6b0926bac70f70f467394293e4b11bb2f23` |
| selected epoch | 9 (best val macro F1 0.4530) |

Downstream ablation-matrix seed variation (42/1337/2024) comes from the
new fusion/relational/shift module's initialization and data order, not
from re-encoding text — this one checkpoint is used for every config and
every seed in the matrix.

### Cache build (`scripts/build_text_ctx_cache.py`)

Caches `ContextTextClassifier.encode(...)` (masked mean of
`last_hidden_state`, the same k=8 context-window construction as Phase T,
`src/rapport/models/text_classifier.py`'s pooling logic exposed as a
separate `encode` method) to
`data/meld/cache/text_ctx/{split}/dia{d}_utt{u}.pt`.

**Real bug found and fixed before this cache was trusted:** the first
version of the build script wrote cache filenames using the *positional*
index within each dialogue (`MELDContextTextDataset.index`'s `t`), not the
raw `utterance_id` column. Verified directly that these differ: 73/1038
train dialogues (3/114 dev, 2/280 test) have non-contiguous `utterance_id`
sequences (gaps from upstream bad-clip exclusion), so positional index
diverges from the true utterance_id partway through any dialogue with a
gap. This surfaced immediately as a `FileNotFoundError` when the
verification script tried to load a cache file by its real utterance_id
(`dia1_utt8.pt` didn't exist — it had been written as a different,
wrong filename). Fixed by looking up the real `utterance_id` value per
row instead of using the positional index, and the cache was rebuilt from
empty (file counts unchanged: 9988/1108/2610, since utterance_id values
are unique within a dialogue either way — only the *naming* was wrong,
not the count).

```
[build_text_ctx_cache] split=train wrote 9988 files in 9.9s
[build_text_ctx_cache] split=dev wrote 1108 files in 1.2s
[build_text_ctx_cache] split=test wrote 2610 files in 2.8s
```

`manifest.json`: `cache_version=text_ctx_v1`, checkpoint path + sha256
recorded, `k=8`, `max_length=256`, `pooling=masked_mean_last_hidden_state`.

### Process-independence check (`scripts/verify_text_ctx_cache.py`)

21 samples across all 3 splits, recomputed in a fresh process (fresh model
instance, fresh tokenizer) and compared to the cached tensor: **21/21
`allclose` (atol/rtol 1e-4), overall max abs diff 0.000010** — floating-point
noise, not a real mismatch.

### Sanity probe

`LogisticRegression` on cached `text_ctx` train embeddings, evaluated on
test:

| metric | value |
|---|---|
| weighted F1 | **0.6119** |
| macro F1 | 0.4191 |

Within the expected ~0.62-0.64 range (a fresh linear head naturally trails
the encoder's own fine-tuned classifier head slightly — Phase T's
end-to-end raw weighted F1 was 0.6403 — but this confirms the cached
embeddings carry strong, intact signal). This is the new context-aware
feature ceiling reference for Phase N4's fusion configs.

**Step 0 complete.** Next: Step 1, `base_fusion`.

---

## STEP 1 — `base_fusion` (the ablation matrix's internal baseline)

`src/rapport/models/rapport_model.py`'s `RapportModel` (all flags off):
`U_e = ReLU(W_proj[text_ctx||A||V]+b)` — structurally identical to
`SocialGNN`'s existing fusion layer, just fed the Phase T `text_ctx` cache
instead of the old frozen non-contextual text cache — then the unchanged
v1-style per-speaker GAT+GRUCell node update and classifier
(`rapport.models.social_gnn.GraphAttentionLayer` reused directly, not
reimplemented). `MELDCachedDataset` gained a `text_cache_subdir` param
(default `"text"`, set to `"text_ctx"` here) so both text caches remain
available side by side.

**Recipe exceptions for Phase N4** (documented, not a silent change):
loss is plain CE (`nn.CrossEntropyLoss`, no focal/tempered-alpha), and
checkpoint selection / early stopping use **val macro F1**, not
RECIPE.md's original val weighted F1 convention. A frozen post-hoc
tau=0.25 adjustment (from Phase T) is applied at eval time as a secondary
reported score only, never for selection. Optimizer (AdamW lr=1e-4),
cosine schedule, dropout 0.5, hidden dim 256, max 100 epochs, early stop
patience 10, bf16 — all unchanged from RECIPE.md's locked GNN values.

Unit tests (`tests/test_rapport_model.py`, 4 tests): forward is finite
with audio/video zeroed (text_ctx-only pathway); the model actually
trains and perfectly fits a tiny synthetic batch under that same
zeroed-A/V condition; bit-for-bit reproducibility given a fixed seed;
`relational=True`/`shift=True`/`temporal=True` raise `NotImplementedError`
until their respective steps land.

### `base_fusion` seed 42 result

`docs/train_base_fusion_seed42.log`, `outputs/base_fusion_seed42/metrics.json`.
Early-stopped at epoch 22 (best: epoch 12, avg 7.96s/epoch, ~3 min wall
clock).

| metric | raw | tau_eval=0.25 |
|---|---|---|
| weighted F1 | 0.6345 | 0.6393 |
| macro F1 | 0.4359 | 0.4601 |
| accuracy | 0.6533 | — |
| neutral | 0.796 | 0.792 |
| joy | 0.611 | 0.611 |
| sadness | 0.355 | 0.383 |
| anger | 0.460 | 0.472 |
| surprise | 0.578 | 0.579 |
| fear | 0.065 | 0.158 |
| disgust | 0.186 | 0.226 |
| all 7 nonzero | yes | yes |

`base_fusion`'s raw weighted F1 (0.6345) lands essentially at the Phase T
text-only anchor (0.6403 mean) — consistent with text_ctx being the
dominant modality and audio/video contributing comparatively little at
this fusion depth, matching every prior linear-probe finding in this
project (`docs/DIAGNOSIS.md`: text >> audio > video). This is the
internal baseline the relational/shift/temporal ablations will be
measured against, not yet a result to gate on (Step 6 does that with all
3 seeds of the `full` config).

**Step 1 complete.** Next: Step 2, relational edge memory.

---

## STEP 2 — Relational edge memory

`src/rapport/models/relational_memory.py` (`pair_index` +
`RelationalEdgeMemory`) implements spec section A exactly: canonical
triangular pair indexing, one shared `GRUCell` for incident-only edge
updates, GATv2-style edge-conditioned attention/messages, plain-mean
readout term. Wired into `RapportModel._forward_relational`
(`src/rapport/models/rapport_model.py`); the non-relational path
(`_forward_base`) is untouched, byte-for-byte the same code as Step 1.

**Wiring decision the spec left open** (recorded in full in
`docs/SPEC_RAPPORT_COMPONENTS.md`'s Implementation decisions): the
edge-conditioned attention context feeds the node GRU's **input** (concatenated
with the utterance embedding), while the speaker's own raw previous state
stays the GRU's **hidden** argument — keeping "my own history" and "what
I'm picking up about my relationships" as separate signals rather than
pre-mixed, since edges carry no self-information to begin with (no i=i
pairs). Classifier input dim: `HIDDEN_DIM` (256) for `relational=False`,
`HIDDEN_DIM + EDGE_DIM` (256+128=384) for `relational=True` — two separate
`nn.Linear` layers, not a shared one, so A7 has zero risk of an accidental
shape interaction.

### Test coverage (A9)

- Module-level (`tests/test_relational_memory.py`, 11 tests): `pair_index`
  bijectivity/contiguity/symmetry/self-pair rejection; incident-only
  updates leave non-incident edges untouched; implicit-zero-init matches a
  hand-reproduced GRU call; ordering (attend() is sensitive to
  fresh-vs-stale edge values, proving update-then-attend actually
  matters); isolated-speaker zero fallbacks (attend, edge_mean); edge_mean
  averages only incident edges; N=2 dyadic case; gradient flow to every
  parameter.
- Model-level (`tests/test_rapport_model_relational.py`, 5 tests): finite
  forward; N=2 dyadic end-to-end; **A7** — `relational=False`'s public
  `forward()` output is bit-for-bit `torch.equal` to a direct call to
  `_forward_base` on the same fixed input/weights; per-dialogue reset — two
  different dialogues in a batch that happen to reuse the same *local*
  speaker-id numbering (0, 1) produce independent outputs whether run
  alone or batched together (`allclose`, same isolation pattern
  `docs/DIAGNOSIS.md`'s graph-state-isolation test used for the original
  `SocialGNN`); gradient flow to every parameter, relational and shared.

All 20 tests pass (11 + 5, plus 4 carried over from Step 1's
`test_rapport_model.py`).

### Diagnostic run — relational-only, seed 42 (not a final ablation-matrix cell)

Not one of the 5 configs in Step 5's matrix (`full` /
`minus_relational` / `minus_shift` / `minus_temporal` / `base_fusion` —
all of which pair relational with the not-yet-implemented shift/temporal
flags except `minus_shift`/`minus_temporal`/`base_fusion`), but run here
purely to confirm the new code path trains end-to-end before moving on,
mirroring how Step 1 needed a real training run to validate `base_fusion`.
`docs/train_relational_only_seed42_diagnostic.log`. ~15s/epoch (vs.
`base_fusion`'s ~8s/epoch — expected, from the per-active-speaker Python
loop the variable neighbor-count attention requires).

Early-stopped at epoch 19 (best: epoch 9, avg 14.86s/epoch, ~5 min wall
clock — roughly 2x base_fusion's per-epoch time, from the per-active-speaker
Python loop the variable-neighbor-count attention requires; no batching
optimization attempted, not currently a bottleneck at this scale).

| metric | raw | tau_eval=0.25 |
|---|---|---|
| weighted F1 | 0.6383 | 0.6374 |
| macro F1 | 0.4348 | 0.4329 |
| accuracy | 0.6559 | — |
| all 7 nonzero | yes | yes |

Comparable to `base_fusion`'s seed-42 result (raw weighted F1 0.6345,
macro F1 0.4359) — relational memory alone, on a single seed, isn't
dramatically better or worse here. Expected: this diagnostic isn't a fair
test of the relational component's value (that's Step 6's `full` vs
`minus_relational` 3-seed delta, with shift and temporal also present);
it exists purely to confirm the code path trains end-to-end without
pathology before moving on, which it does.

**Step 2 complete.** Next: Step 3, shift objective.

---

## STEP 3 — Emotion-shift auxiliary objective

**B1/B2 (labels + stats):** `rapport.data.shift_labels.add_shift_labels`
(7 unit tests) derives `shift_label`/`shift_mask` into the processed
parquets; `docs/SHIFT_LABEL_STATS.md` confirms the method's premise
cleanly across all 3 splits (no flag raised) — minority classes'
eligible-row shift rates are 1.5-2.1x neutral's (e.g. test split: neutral
0.39 vs fear 0.92, disgust 0.85, surprise 0.76).

**B3/B4 (model + loss):** `RapportModel(shift=True)` adds
`nn.Linear(HIDDEN_DIM, 1)` applied to the raw node state `h_{s,t}` (not
the edge-augmented readout the emotion classifier sees), in both the
relational and non-relational paths — `forward()` now always returns
`(emotion_logits, shift_logits)`, `shift_logits=None` when `shift=False`
(recorded in `docs/SPEC_RAPPORT_COMPONENTS.md`'s Implementation
decisions). `scripts/train_rapport.py`: `BCEWithLogitsLoss(pos_weight=...)`
computed from the TRAIN split's shift rate (0.7928), combined as
`L = L_CE_emotion + 0.5 * L_shift` with masked positions (batch padding
AND each speaker's first-in-dialogue utterance) excluded via boolean
indexing before the loss call.

**B5 (selection):** unchanged — checkpoint selection and early stopping
remain on EMOTION val macro F1 only; shift F1 is computed and logged every
epoch (`val_shift_f1` in `history`) purely as an auxiliary metric.

**B6 (tests):** `tests/test_shift_labels.py` (7, label derivation + toy
dialogue + pos_weight) + `tests/test_rapport_model_shift.py` (5: shift
logits shape/None-when-disabled, works with `relational=True` too, masked
positions contribute exactly zero gradient — verified by flipping a
masked position's label and confirming bit-identical loss/gradients with
dropout disabled to remove an unrelated source of run-to-run noise — and
the combined CE+shift loss trains a tiny synthetic batch to perfect
emotion accuracy). All pass.

### Diagnostic run — shift-only, seed 42 (not a final ablation-matrix cell)

`docs/train_shift_only_seed42_diagnostic.log`, run purely to confirm the
combined-loss training loop is sane end-to-end (mirrors Step 2's
relational-only diagnostic). Early-stopped at epoch 30 (best: epoch 20,
avg 9.16s/epoch — essentially the same speed as `base_fusion`, since the
shift head is a single small linear layer).

| metric | raw | tau_eval=0.25 |
|---|---|---|
| weighted F1 | 0.6370 | 0.6369 |
| macro F1 | 0.4542 | **0.4707** |
| accuracy | 0.6460 | — |
| fear | (see per-class) | **0.268** |
| disgust | (see per-class) | 0.208 |
| all 7 nonzero | yes | yes |
| test shift F1 (aux, not selected on) | — | 0.6370-ish range during training, not separately reported at test in this diagnostic |

Notably stronger than `base_fusion` (raw macro F1 0.4359, adjusted macro
F1 0.4601) and the relational-only diagnostic (0.4348 / 0.4329) on this
single seed — fear F1 in particular (0.268 adjusted) is the best of any
Phase N4 diagnostic so far, consistent with `docs/SHIFT_LABEL_STATS.md`'s
finding that fear has the single highest shift rate (0.84-0.92 across
splits) of any class. Not dispositive on its own (single seed, single
component), but a promising early signal for Step 6's full ablation.

**Step 3 complete.** Next: Step 4, temporal attention.

---

## STEP 4 — Temporal attention pooling

`src/rapport/models/temporal_attention.py`'s `TemporalAttentionPool`
implements spec section C: a learnable aggregation token, one block of
4-head self-attention, output = the aggregation token's position after
the block.

**Two implementation decisions the spec left open** (full detail in
`docs/SPEC_RAPPORT_COMPONENTS.md`):
1. **Mean-pool-equivalent init (C2):** a literal post-residual LayerNorm
   can never exactly reproduce a raw masked mean (its normalization has no
   learnable identity setting), so LayerNorm is applied to the query/key
   path only ("QK-norm"), never to values or the residual sum. Combined
   with zero-initializing the query projection and identity-initializing
   the value/output projections, the block's output is EXACTLY
   `agg_token + masked_mean(tokens)` at init (`agg_token` itself zero-init)
   — verified to atol 1e-5, not just approximately, in
   `tests/test_temporal_attention.py`.
2. **Separate pool per modality (C4):** `video_temporal_pool` and
   `audio_temporal_pool` are two independent instances, not shared
   weights — audio (wav2vec2) and video (MViTv2) tokens come from
   different frozen backbones with different statistics.

**Data pipeline:** `MELDCachedDataset(load_av_tokens=True)` additionally
loads the pre-pooling `video_tokens`/`audio_tokens` caches (video: fixed
392 tokens/utterance, no inner padding needed; audio: variable length, up
to 64, `AUDIO_TOKEN_STRIDE`/`AUDIO_TOKEN_MAX_LEN`-capped).
`collate_dialogues` gained `_pad_token_sequences` (mirrors the existing
`_pad_waveforms`/`_pad_token_ids` two-level padding pattern) producing
`[B, L, T_max, 768]` + a `[B, L, T_max]` boolean mask per modality.

**A real correctness hazard found and handled, not just noted:**
batch-padding dialogue positions (from a shorter dialogue sharing a batch
with a longer one) have an all-False token mask — feeding that straight
into the pool's softmax divides 0/0 (NaN), which would then contaminate
every OTHER row's gradient for the pool's shared parameters once summed
during backward. `RapportModel._temporal_pool_av` filters to only the
`dialogue_mask`-real positions before pooling and scatters the result
back into an all-zero tensor via `index_copy` (differentiable, unlike
in-place indexed assignment) — covered by a dedicated test
(`test_temporal_handles_batch_padding_without_nan`) that deliberately
constructs a mixed-length batch and checks both the forward output and
every parameter's gradient are finite.

### Test coverage (C3 + integration)

- Module-level (`tests/test_temporal_attention.py`, 6 tests): output
  matches the masked mean at init to atol 1e-5, with and without padding;
  padded positions provably don't influence the output (corrupting them
  with huge noise changes nothing), both at init and after training (to
  confirm the masking discipline isn't an init-only accident); gradient
  flows to every parameter; the module is trainable (fits a random
  regression target).
- Model-level (`tests/test_rapport_model_temporal.py`, 4 tests): finite
  forward; works combined with `relational=True` and `shift=True`
  simultaneously; the batch-padding/NaN hazard above; gradient flow to
  the temporal-pool parameters specifically.

All 10 pass, plus every earlier test still passes (74 total across the
whole Phase N4 + Phase T test suite).

### Diagnostic run — temporal-only, seed 42 (not a final ablation-matrix cell)

`docs/train_temporal_only_seed42_diagnostic.log`. ~17s/epoch (slower than
`base_fusion`'s ~8s — loading and attending over token sequences, mostly
audio's variable-length padding, costs more than reading one pre-pooled
768-d vector per utterance). Early-stopped at epoch 16 (best: epoch 6).

| metric | raw | tau_eval=0.25 |
|---|---|---|
| weighted F1 | 0.6328 | 0.6379 |
| macro F1 | 0.4150 | 0.4455 |
| accuracy | 0.6567 | — |
| all 7 nonzero | yes | yes |

Slightly weaker than `base_fusion` alone on this single seed (raw macro F1
0.4150 vs. 0.4359) — not concerning: video/audio have consistently been
the weakest signal sources throughout this project (`docs/DIAGNOSIS.md`'s
linear probes: text >> audio > video), so a better pooling mechanism for
those two modalities alone, with nothing else changed, isn't guaranteed to
move the needle much on a single seed. This diagnostic's purpose was
confirming the code path (including the padding/NaN hazard fix) is
correct end-to-end, which it is.

**Step 4 complete.** All four spec components (base fusion, relational
memory, shift objective, temporal attention) are implemented and
unit-tested (74 tests pass across the whole `tests/` suite, including
`tests/test_rapport_model*.py`, `tests/test_relational_memory.py`,
`tests/test_shift_labels.py`, `tests/test_temporal_attention.py`, and
every earlier Phase T test), and each new component individually verified
to train end-to-end via a real diagnostic run. Next: Step 5, the full
ablation matrix.

---

## STEP 5 — Ablation matrix (15 runs)

`scripts/run_ablation_matrix.py`: idempotent runner over
`{full, minus_relational, minus_shift, minus_temporal, base_fusion} x
seeds {42, 1337, 2024}`, skipping any `(config, seed)` whose
`outputs/<run_name>/metrics.json` already exists —
`base_fusion_seed42` (Step 1) was correctly reused rather than retrained.
Launched via `nohup` + log-poll, `docs/train_ablation_matrix.log`.

First run (`full_seed42`, the heaviest config) took 691.5s (~11.5 min);
projected total for the remaining 14 runs was ~2-3 hours, reported and
proceeded per instructions (well under the 24h budget). **Actual total:
all 15 runs (14 executed + 1 reused) completed in that range**, no
crashes, no OOM, no manual intervention needed.

**Step 5 complete.** Next: Step 6, gates and final reporting.

---

## STEP 6 — Gates and final reporting

Full gate check, main table, component-attribution view, and per-seed
appendix: **`docs/PHASE_N4_STEP6.md`** (auto-generated by
`scripts/report_ablation_matrix.py` from all 15 `metrics.json` files, with
a hand-written synthesis section).

**Headline result: the HARD gate FAILS, cleanly and unambiguously.**
`base_fusion` (none of RAPPORT's three components) is the single
best-performing config of all five; `full` (all three) is the
second-worst, below the text-only anchor by 0.0175 and below its own
`base_fusion` baseline by 0.0129. The pattern is monotonic and consistent
across weighted F1, macro F1, and the rare-class (fear/disgust) means the
shift objective was specifically meant to help. Full detail, including
why this isn't just a marginal miss, is in `docs/PHASE_N4_STEP6.md`'s
Synthesis section.

**Per instructions and this project's established practice at every
prior gate failure: report completely, commit, push, STOP.** No fix is
proposed or applied here.

**PHASE N4 COMPLETE (gate: FAILED).**
