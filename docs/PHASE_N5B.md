# PHASE N5-B — IEMOCAP: the relational hypothesis's fair trial

**STATUS: BLOCKED.** Checked this machine directly before starting
anything: no IEMOCAP data, no download script, and no credentials exist
here -- only a declarative stub (`configs/data/iemocap.yaml`, `num_classes:
6`, empty `label_map`). The phase instructions themselves note this is
blocked on credentials ("human is checking the TUIT inbox") and require
waiting for an explicit choice on the label-protocol decision point (4-class
vs. 6-class) before preprocessing -- a choice that can't be made without the
raw label distribution in hand, which requires the data. Nothing in this
phase was attempted beyond confirming the blocker; no placeholder
preprocessing, no guessed credentials, no fabricated label counts.

## What's needed to unblock

1. IEMOCAP access (credentials/download) -- outside this session's control.
2. Once data is available: preprocessing per the original Phase 7 spec
   (session structure, 16kHz audio, 8-frame video, transcripts), then
   **present the 4-class vs. 6-class label counts and wait for the
   protocol choice** before building the processed index. Session5=test,
   Session4=val, stated explicitly per the phase instructions.
3. Phase T text encoder retrained on IEMOCAP (same k=8, locked recipe,
   `docs/RECIPE.md`) as a new, IEMOCAP-specific anchor -- the MELD Phase T
   checkpoint does not transfer (different label set, different domain).
4. A/V caches, then the residual matrix `{base_fusion_R, full_R,
   minus_relational_R} x 3 seeds`.
5. Pre-registered hypothesis (stated here, before any IEMOCAP result
   exists, so it can't be fitted after the fact): **relational memory's
   delta on IEMOCAP > its delta on MELD, driven by dialogue length**
   (IEMOCAP dialogues run substantially longer than MELD's, which should
   give graph-level relationship state more room to matter than MELD's
   `docs/PHASE_N4R.md`/`docs/PHASE_N5A.md` results showed). Either outcome
   -- confirmed or not -- gets reported when this phase actually runs.
   Also log the single dyadic edge-state norm over time for 3 sample
   sessions as a qualitative figure candidate, once relational memory is
   running on real IEMOCAP dialogues.

**Nothing further happens on N5-B until IEMOCAP access exists.**
