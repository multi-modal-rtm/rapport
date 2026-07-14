# PHASE T — contextual text encoder (foundation rebuild)

Goal: replace isolated-utterance frozen text features with a context-window,
LoRA-tuned RoBERTa trained end-to-end. Text-only — no fusion, no GNN, no
video/audio. This phase certifies the text foundation before anything is
built on top of it.

## 1. Input construction — `src/rapport/data/context_text.py`

For utterance t: `"<spk_{t-k}>: u_{t-k} </s> ... <spk_{t-1}>: u_{t-1} </s>
<spk_t>: u_t"`, using the k most recent PRECEDING utterances in the same
dialogue (never future utterances). Speaker names are the raw MELD speaker
strings as plain-text prefixes (e.g. `"Chandler:"`), `k=8` default,
`max_length=256`. Segments are joined with the tokenizer's own `sep_token`
(`</s>`) so it's encoded as the real special-token id rather than literal
characters (verified directly: it round-trips through `tokenizer.decode`
correctly). If the full window doesn't fit `max_length`, context segments
are dropped OLDEST-first, one at a time, until it does; the current
utterance (the last segment) is never itself a drop candidate and is only
truncated as a last resort if it alone still overflows (has not occurred in
practice — MELD utterances are short sitcom lines).

Batching is utterance-level in this phase (`MELDContextTextDataset`
flattens every dialogue into independent, randomly-shufflable examples) —
there's no recurrent/graph state to preserve dialogue-level batching for,
unlike `rapport.data.meld`'s dialogue-grouped datasets used by the GNN
architectures.

**Real environment bug found and worked around:** on this machine's
`transformers` version, constructing `RobertaTokenizerFast` directly leaves
`backend_tokenizer.pre_tokenizer`/`decoder` unset, silently falling back to
character-level tokenization with no byte-level BPE merges (verified
directly: `RobertaTokenizerFast()('Hello world')` produces 10 single-character
tokens instead of the expected `['Hello', 'Ġworld']`). `AutoTokenizer.from_pretrained("roberta-base")`
resolves to a correctly configured fast tokenizer and is used everywhere in
this phase instead. This does not affect the existing frozen-backbone code
(`rapport.models.backbones.RobertaBackbone`) or its cached features, which
never decode text and were unaffected by this specific bug in the way that
mattered for their use case — but it's a live footgun for any future code in
this repo that reaches for `RobertaTokenizerFast` directly.

Unit tests (`tests/test_context_text.py`, 9 tests, all pass): no future
leakage, first utterance of a dialogue has no context, truncation preserves
the current utterance intact, determinism, collator padding/masking.

## 2. Model — `src/rapport/models/text_classifier.py`

`roberta-base` + LoRA (`peft`, r=8, alpha=16, dropout=0.05, target
`["query", "value"]`) + a linear classification head on the masked mean of
`last_hidden_state`. Trainable parameters are exactly the LoRA adapters +
head; every other RoBERTa parameter is frozen by `get_peft_model`.
Trainable-param budget is asserted (`< 5%` of total) and logged:

```
trainable params: 296,199 / 124,351,239 (0.2382%)
```

Unit tests (`tests/test_text_classifier.py`, 4 tests, all pass): forward
shape, only LoRA+head require grad, trainable-param budget, gradients flow
only to trainable params.

## 3. Loss — logit adjustment (Menon et al. 2021)

`src/rapport/training/losses.py`: `compute_class_priors` (train-split label
frequency) + `LogitAdjustedLoss`, which subtracts `tau * log(prior_c)` from
the logits before standard cross-entropy (`tau=1.0`). At eval, raw
(unadjusted) logits are used directly for argmax/scoring — `evaluate_split`
in the training script never applies the adjustment. This supersedes focal
loss for this phase only; `docs/RECIPE.md` now has a scope note explaining
the locked recipe (focal + tempered alpha) governs the GNN architectures
and Phase T is a documented exception (CE + LA). 5 new unit tests, all pass.

## 4. Training — `scripts/train_context_text.py`

AdamW, lr=2e-4 (LoRA+head only), linear warmup over the first 10% of steps
then linear decay, max 10 epochs, early stop patience 3 on **val macro F1**
(both checkpoint selection and early stopping use macro F1 — unlike the
locked GNN recipe, which uses val weighted F1), bf16 autocast, batch by
utterance (batch size 32), grad clip 1.0. Standalone script, not the Hydra
`rapport.__main__` pipeline (this phase predates any GNN/fusion
integration).

## 5. GATE — seed 42 result

Run: `docs/train_context_text_seed42.log`, `outputs/context_text_seed42/metrics.json`.
Early-stopped at epoch 7 (patience 3), best checkpoint epoch 4 (avg
11.4s/epoch, ~85s total wall clock).

| metric | value |
|---|---|
| test weighted F1 | **0.6041** |
| test macro F1 | 0.3678 |
| test accuracy | 0.6418 |
| neutral F1 | 0.793 |
| joy F1 | 0.584 |
| sadness F1 | 0.243 |
| anger F1 | 0.429 |
| surprise F1 | 0.525 |
| fear F1 | **0.000** |
| disgust F1 | **0.000** |

### Pre-registered gate: seed 42 test weighted F1 >= 0.60 AND all 7 classes nonzero

| criterion | required | actual | pass? |
|---|---|---|---|
| weighted F1 | >= 0.60 | 0.6041 | **yes** |
| all 7 classes nonzero | 7/7 | 5/7 (fear, disgust = 0) | **NO** |

**GATE: FAILED** — on the classes-nonzero criterion only; weighted F1
clears the bar comfortably (+0.0041).

### Why this doesn't fit the phase's three pre-registered branches

The instructions specify three outcomes keyed off the weighted-F1 value:
"if passed" (both criteria) -> 3-seed table; "[0.57, 0.60)" -> k-ablation;
"below 0.57" -> stop, report curves, don't iterate. **0.6041 lands in none
of these** — it's >= 0.60 (not in the ablation band, not below the stop
floor) but the conjunction still fails because of the untested
classes-nonzero criterion. This is flagged rather than resolved
unilaterally, per this project's established pattern (`docs/DIAGNOSIS.md`'s
gate-failure investigations) of not inventing new remediations for a
gate outcome the pre-registration didn't anticipate.

### One relevant, non-dispositive observation from the training curve

Looking at `metrics.json`'s per-epoch val history: disgust's val F1 was
0.000 through epoch 4 (the selected checkpoint) but went to a small nonzero
value (0.087) at epochs 5, 6, and 7 — all of which scored *lower* on val
macro F1 than epoch 4 (0.348–0.354 vs. epoch 4's 0.380) because of trade-offs
elsewhere (anger/sadness dropped). Fear's val F1 was 0.000 at every logged
epoch, 0 through 7 — not a selection artifact, consistent with fear simply
not being learned within this training budget. This is the same
disgust-collapse-under-a-fixed-selection-metric shape documented for the
GNN architecture in `docs/DIAGNOSIS.md`'s original gate-failure
investigation, now recurring in the text-only foundation model — noted for
context, not acted on (no checkpoint-selection change has been made; that
would be exactly the kind of unregistered iteration the phase instructions
say not to do below 0.57, and this case isn't even below 0.57).

**Per instructions, seeds 1337/2024 are NOT run and the k-ablation is NOT
run** — neither pre-registered trigger condition is met. Stopping here and
flagging back for direction on how to treat a weighted-F1 pass combined
with a classes-nonzero fail, rather than deciding unilaterally.
