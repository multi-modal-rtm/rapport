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
