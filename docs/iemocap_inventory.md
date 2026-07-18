# IEMOCAP corpus inventory + label-protocol audit (Phase N5-B, Step B1)

Read-only inventory only -- no preprocessing, no processed index, no caches. Release: `/home/devops/data/iemocap/iemocap/IEMOCAP_full_release` (md5 521be1e5eec425ae21fdc27c763ca813, verified before extraction). **Location note:** this is outside the repo, not at `configs/data/iemocap.yaml`'s expected `${paths.data_dir}/iemocap` (`data/iemocap/`) -- flagged for whoever builds the processed index later; nothing moved or symlinked here.

## Pre-registered hypothesis (docs/PHASE_N5B.md, commit 04d9267, quoted verbatim -- predates this data download)

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

## 1. Corpus inventory

| session | dialogues (wav) | improvised | scripted | avi | transcription | EmoEvaluation |
|---|---|---|---|---|---|---|
| Session1 | 28 | 14 | 14 | 28 | 28 | 28 |
| Session2 | 30 | 16 | 14 | 30 | 30 | 30 |
| Session3 | 32 | 18 | 14 | 32 | 32 | 32 |
| Session4 | 30 | 16 | 14 | 30 | 30 | 30 |
| Session5 | 31 | 16 | 15 | 31 | 31 | 31 |
| **total** | **151** | | | | | |

**Utterances (EmoEvaluation consensus turns): 10039** (published expectation: ~10039 -- MATCHES exactly).

**Dialogues: 151** (published expectation: 151 -- MATCHES exactly).

**Distinct (session, gender) speaker slots: 10** (published expectation: 10 -- MATCHES exactly); IEMOCAP's dyadic design assigns one male + one female actor per session, consistent across sessions except that the SAME 10 individuals are not necessarily distinct across sessions in the original corpus design (verify against Documentation/ if a specific speaker-identity check is later needed -- not required for this inventory).

**No missing or unreadable dialogue-level files** -- wav/avi/transcription/EmoEvaluation all present for every dialogue in every session.


## 2. Label audit -- raw distribution (all categories, incl. xxx/oth)

| emotion code | Session1 | Session2 | Session3 | Session4 | Session5 | total |
|---|---|---|---|---|---|---|
| ang | 229 | 137 | 240 | 327 | 170 | 1103 |
| dis | 1 | 0 | 1 | 0 | 0 | 2 |
| exc | 143 | 210 | 151 | 238 | 299 | 1041 |
| fea | 12 | 9 | 2 | 7 | 10 | 40 |
| fru | 280 | 325 | 382 | 481 | 381 | 1849 |
| hap | 135 | 117 | 135 | 65 | 143 | 595 |
| neu | 384 | 362 | 320 | 258 | 384 | 1708 |
| oth | 1 | 1 | 0 | 1 | 0 | 3 |
| sad | 194 | 197 | 305 | 143 | 245 | 1084 |
| sur | 25 | 17 | 28 | 19 | 18 | 107 |
| xxx | 415 | 436 | 572 | 564 | 520 | 2507 |
| **total** | 1819 | 1811 | 2136 | 2103 | 2170 | 10039 |

## 3. Protocol comparison: 6-class vs. 4-class

- **6-class** {angry, happy, excited, sad, neutral, frustrated} (codes: {ang, hap, exc, sad, neu, fru})
- **4-class** {angry, happy+excited, neutral, sad} (codes: {ang, hap, exc, sad, neu}, happy+excited merged)

### Per-session utterance counts kept, by protocol

| session | 6-class kept | 4-class kept | total in session |
|---|---|---|---|
| Session1 | 1365 | 1085 | 1819 |
| Session2 | 1348 | 1023 | 1811 |
| Session3 | 1533 | 1151 | 2136 |
| Session4 | 1512 | 1031 | 2103 |
| Session5 | 1622 | 1241 | 2170 |
| **total** | **7380** | **5531** | **10039** |

**Discarded utterances:** 6-class discards **2659** (26.5%); 4-class discards **4508** (44.9%).

### Session4 (val) / Session5 (test) specifically

| split | session | 6-class kept | 4-class kept | total in session |
|---|---|---|---|---|
| val | Session4 | 1512 | 1031 | 2103 |
| test | Session5 | 1622 | 1241 | 2170 |

### Per-class counts (kept utterances only)

**6-class:**

| class | count |
|---|---|
| frustrated | 1849 |
| neutral | 1708 |
| angry | 1103 |
| sad | 1084 |
| excited | 1041 |
| happy | 595 |

**4-class:**

| class | count |
|---|---|
| neutral | 1708 |
| happy+excited | 1636 |
| angry | 1103 |
| sad | 1084 |

### Literature comparability (one sentence per protocol, no further recommendation)

- **4-class** {angry, happy+excited, sad, neutral} is the more common split in the emotion-recognition literature (comparable to most published IEMOCAP SOTA tables, which pool happy+excited and drop frustrated/fear/disgust/surprise/xxx/other).
- **6-class** {angry, happy, excited, sad, neutral, frustrated} keeps happy and excited separate and adds frustrated -- less common as a direct comparison point, but closer to MELD's own 7-class granularity (docs/PHASE_T.md) if cross-corpus label-set similarity to this project's MELD work is the priority instead of literature comparability.

## STOP -- protocol choice is not made here

Per the B1 instructions: counts presented above, no preprocessing attempted, no processed index built. Waiting on the 4-class vs. 6-class decision before any further N5-B work.