# Emotion-shift auxiliary label statistics

`scripts/add_shift_labels.py` (logic in `src/rapport/data/shift_labels.py`,
7 unit tests) derived `shift_label`/`shift_mask` into
`data/meld/processed/{train,dev,test}.parquet`, per
`docs/SPEC_RAPPORT_COMPONENTS.md` section B1: `shift_label=1` iff an
utterance's emotion label differs from the SAME SPEAKER's previous
utterance label within the dialogue; a speaker's first utterance in a
dialogue is excluded (`shift_mask=0`).

## Eligibility and overall shift rate

| split | rows | eligible (shift_mask=1) | eligible % | overall shift rate | pos_weight |
|---|---|---|---|---|---|
| train | 9988 | 7173 | 71.8% | 0.5578 | 0.7928 |
| dev | 1108 | 765 | 69.0% | 0.5556 | 0.8000 |
| test | 2610 | 1864 | 71.4% | 0.5381 | 0.8584 |

(`pos_weight` = eligible-negative-count / eligible-positive-count, for
`BCEWithLogitsLoss` per spec B3 — computed here per split for reference;
the training script uses the TRAIN split's value only, per spec B3's
"pos_weight from the train-split shift rate".)

## Per-class shift rate (among eligible rows)

| emotion | train n | train rate | dev n | dev rate | test n | test rate |
|---|---|---|---|---|---|---|
| neutral | 3314 | 0.3956 | 309 | 0.4304 | 877 | 0.3877 |
| joy | 1178 | 0.6503 | 99 | 0.6970 | 273 | 0.6777 |
| sadness | 522 | 0.7414 | 88 | 0.6364 | 161 | 0.6522 |
| anger | 888 | 0.6227 | 119 | 0.4958 | 265 | 0.5434 |
| **surprise** | 857 | **0.7503** | 103 | **0.7184** | 205 | **0.7610** |
| **fear** | 208 | **0.8413** | 32 | **0.6875** | 37 | **0.9189** |
| **disgust** | 206 | **0.8058** | 15 | **0.8000** | 46 | **0.8478** |

## Premise check: do minority classes (fear, disgust, surprise) show elevated shift rates vs neutral?

**Yes, clearly, in every split — no flag raised.** Neutral sits at
0.39-0.43 shift rate in all 3 splits; fear, disgust, and surprise are all
at least ~1.5x that (surprise) to ~2.1x (fear, test split) higher, with no
exception across train/dev/test. This is exactly the method's premise: a
speaker's rare-emotion utterances are disproportionately *transitions*
away from their own preceding emotional state (often neutral), rather than
sustained states — the auxiliary shift objective (spec B) has real,
consistent signal to learn from for precisely the classes the emotion
classifier struggles with most (fear/disgust's collapse-to-zero failure
mode, documented repeatedly in `docs/DIAGNOSIS.md` and `docs/PHASE_T_DIAGNOSIS.md`).

Non-neutral classes are all elevated relative to neutral, not just the
three flagged as "minority" — joy (0.65-0.70) and anger (0.50-0.62) are
also above neutral, just by a smaller margin than fear/disgust/surprise.
Sadness is a partial exception in dev (0.64, closer to anger than to
surprise/fear/disgust) but still clearly above neutral in every split.
