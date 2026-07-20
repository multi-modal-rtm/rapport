# IEMOCAP emotion-shift auxiliary label statistics

Companion to `docs/SHIFT_LABEL_STATS.md` (MELD), computed the identical
way (`shift_label`/`shift_mask` columns already present in
`data/iemocap/processed/{train,val,test}.parquet`, derived by
`scripts/preprocess_iemocap.py` reusing `add_shift_labels`
unchanged — same algorithm as MELD: `shift_label=1` iff an utterance's
emotion differs from the SAME SPEAKER's previous utterance within the
dialogue; a speaker's first utterance in a dialogue is excluded
(`shift_mask=0`)). This doc is a read-only pandas aggregation over
already-stored columns — no training, no model involved.

## Eligibility and overall shift rate

| split | rows | eligible (shift_mask=1) | eligible % | overall shift rate | pos_weight |
|---|---|---|---|---|---|
| train | 4246 | 4066 | 95.8% | 0.2838 | 2.5234 |
| val | 1512 | 1452 | 96.0% | 0.2879 | 2.4737 |
| test | 1622 | 1560 | 96.2% | 0.2615 | 2.8235 |

(Eligible % is far higher than MELD's 69–72% — IEMOCAP dialogues are
strictly dyadic, so a speaker's OWN previous utterance is almost always
close by; MELD's multi-party dialogues have many more first-time-
speaking positions per dialogue. Overall shift rate is also markedly
lower than MELD's ~0.54–0.56 — consistent with IEMOCAP's longer, more
emotionally sustained turns.)

## Per-class shift rate (among eligible rows)

| emotion | train n | train rate | val n | val rate | test n | test rate |
|---|---|---|---|---|---|---|
| neutral | 990 | 0.2687 | 237 | 0.3713 | 362 | 0.2431 |
| happy | 374 | 0.3396 | 61 | 0.3607 | 141 | 0.3617 |
| sad | 674 | 0.1944 | 134 | 0.2313 | 231 | 0.1515 |
| angry | 594 | 0.3081 | 321 | 0.3178 | 169 | 0.3728 |
| excited | 470 | 0.2255 | 228 | 0.1491 | 283 | 0.1484 |
| frustrated | 964 | 0.3537 | 471 | 0.2994 | 374 | 0.3449 |

## Premise check: do minority classes show elevated shift rates vs. neutral?

**Mixed — not the clean, exception-free pattern MELD shows.** IEMOCAP's
class distribution is far less skewed than MELD's (the largest class,
`frustrated`, is ~25% of kept utterances vs. MELD's `neutral` at ~48%),
so "minority class" is a weaker frame here to begin with; the closest
analog is `happy` (595 utterances corpus-wide, the smallest of the six).

- **`happy` vs. neutral**: higher in train (0.340 vs.\ 0.269, 1.26$\times$)
  and test (0.362 vs.\ 0.243, 1.49$\times$), but **lower in val** (0.361
  vs.\ 0.371) — one exception out of three splits.
- **`sad` vs. neutral**: LOWER in every split (0.194/0.231/0.152 vs.\
  0.269/0.371/0.243) — the opposite direction from the premise.
- **`excited` vs. neutral**: lower in val and test, roughly tied in train
  — no support for the premise.
- **`frustrated` vs. neutral**: higher in train and test, but **lower in
  val** (0.299 vs.\ 0.371) — one exception.
- **`angry` vs. neutral**: higher in every split (0.308/0.318/0.373 vs.\
  0.269/0.371/0.243) — the one class that cleanly supports the premise
  across all three splits.

**Conclusion for citation purposes**: unlike MELD, where every rare class
showed a consistent, exception-free elevation over neutral, IEMOCAP's
per-class shift rates are inconsistent — only `angry` shows the pattern
cleanly in all three splits; `happy` and `frustrated` each have one
exception (both in val); `sad` and `excited` show the opposite pattern.
The shift-detection auxiliary task's premise is **not confirmed on
IEMOCAP the way it is on MELD** — this should be stated plainly wherever
the premise is cited for both corpora together, not implied to hold
uniformly.
