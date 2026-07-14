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
