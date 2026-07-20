# Draft number audit — Sections 3, 4, 5, 6 (revision round 3)

Re-run after (1) applying `docs/SECTION_6_DRAFT.md`'s pending inserts
([INSERT-A], the 6.1 shift-F1-delta `[CHECK]`), (2) confirming the C5
sentence in 6.4, (3) inserting the SHIFT_LABEL_STATS citation into
Section 3.3, and (4) closing out three leftover `[CHECK]` markers in
Section 4 that were analytically resolved in round 1 but never actually
removed from the draft text (round 2's scope was "verbatim C1–C5 only";
this round's target — "zero mismatches... only open items should be the
four literature `[cite:]` markers... and figure cross-references" — has
no such restriction, so they're closed here).

## Status: target achieved, with two new findings from this round's own verification step

**Zero mismatches remain.** Every open marker across all four drafts is
now either one of the four literature `[cite:]` markers (human task, by
design) or a figure/table cross-reference placeholder (`[EDGE-NORM]`,
`[K-SWEEP]`) awaiting final numbering — exactly the target state.

**Two real, new mismatches were found and fixed during this round's own
verification step**, both self-introduced in my own prior drafting
(round 2's Section 5, and this round's initial pass at filling Section
6's markers) rather than present in `docs/CORRECTIONS_AND_SECTION_5.md`
or `docs/SECTION_6_DRAFT.md`'s original text:

1. **6.2's (and, identically, Section 5's) "eight to ten turns" claim did
   not match the trajectory data precisely.** Task step 5 asked me to
   check this specific claim against the artifact rather than trust the
   earlier report it was written from — checking it directly (turn-to-
   turn increment analysis, `outputs/iemocap_edge_state_trajectory_data.json`)
   showed the steep-rise phase actually ends anywhere from turn ~5 to
   ~12 depending on the dialogue, not a tight 8–10 band. Fixed in both
   `docs/SECTION_5_DRAFT.md` and `docs/SECTION_6_DRAFT.md` to state the
   per-dialogue range precisely instead of a single misleading number.
   The variance-drop claim ("two-to-sevenfold") was also off at the low
   end — actual range is 1.8$\times$–7.4$\times$, not 2–7$\times$ — fixed
   in both files to the precise figures.
2. **6.4's own stated std range (0.005–0.014) contradicted the very
   number in the next clause of the same sentence** (IEMOCAP's
   $\pm0.028$). Checked all six paired-comparison standard deviations
   actually used across the paper (0.0072, 0.0086, 0.0106, 0.0136,
   0.0185, 0.0284) and corrected the stated range to 0.007–0.028, with a
   note on which comparison sits at each end.

## Corrections from round 2, re-confirmed unchanged in this round

C1 (fusion weight norms), C2 ($-0.0048$/$-0.0060$ comparator split), C3
(footnote removed), C4 (1,432 dialogues / 48.1%) — all re-spot-checked
this round, still exact, untouched by this round's edits.

## This round's new insertions/closures, verified

| item | location | value inserted | source | verdict |
|---|---|---|---|---|
| SHIFT_LABEL_STATS citation | §3.3 | MELD: neutral 0.39–0.43 vs.\ fear/disgust/surprise 1.5–2.1$\times$ higher, no exception. IEMOCAP: only partially confirmed (flagged, not glossed) | `docs/SHIFT_LABEL_STATS.md` (unchanged from round 1), `docs/IEMOCAP_SHIFT_LABEL_STATS.md` (new this round) | **match, exact — and honestly reports the IEMOCAP premise is weaker, rather than reusing MELD's clean framing for both corpora** |
| shift-F1 delta `[CHECK]` | §6.1 | MELD $+0.0297$ (0.6682 vs.\ 0.6384); IEMOCAP $+0.0782$ (0.4147 vs.\ 0.3365) | `outputs/{full_R,minus_relational_R}_seed*`, `outputs/{full_R,minus_relational_R}_iemocap_seed*` | **match, exact** — recomputed directly this round, both means of 3 seeds |
| `[INSERT-A]` | §6.1 | full IEMOCAP per-class breakdown (only `angry` confirms cleanly across all 3 splits; `happy`/`frustrated` each have one exception; `sad`/`excited` invert) | `docs/IEMOCAP_SHIFT_LABEL_STATS.md` (new artifact, computed this round from `data/iemocap/processed/*.parquet` — pandas aggregation only, not a training run) | **match — and this is the audit's other headline finding: IEMOCAP's shift-rate premise is genuinely mixed, not a clean confirmation, and the draft now says so explicitly instead of implying uniform support** |
| C5 sentence placement, §6.4 | §6.4 | "Literature comparisons in this paper cite published numbers and are not independently reproduced; in particular, the original implementation of our own prior system was lost to an infrastructure failure..." | `docs/CORRECTIONS_AND_SECTION_5.md` Part 1, C5 | **match, character-for-character** — confirmed present and unaltered |
| leftover §4.1 `[CHECK exact metric label]` | §4.1 | reworded: "balanced" describes training (`class_weight='balanced'`), not a different eval metric — both columns are weighted F1 | `outputs/meld_probe_table.json` (unchanged from round 1's finding) | **resolved this round** (not covered by round 2's C1–C5 list; closed now per this round's unrestricted "zero mismatches" target) |
| leftover §4.3 `[CHECK: minus_relational_R vs full_R deltas]` | §4.3 | weighted F1 $-0.0022$, macro F1 $-0.0023$, fear $-0.0024$, disgust $-0.0084$ (all in `minus_relational_R`'s favor) | `outputs/{full_R,minus_relational_R}_seed{42,1337,2024}/metrics.json` | **resolved this round** — "every aggregate metric" claim now stated with the actual four deltas, all confirming |
| leftover §4.4 `[CHECK: n asymmetry footnote]` | §4.4 | $k=0$ anchor is seed-42-only originally; this consolidation pass added exactly the 4 new seeds requested, not 6 | `docs/PHASE_N5A.md` | **resolved this round** |

## Section 5 — re-confirmed after this round's trajectory-figure fix

All Section 5 numbers from round 2 (hypothesis quote, 7,380/10,039,
1,622/1,241, $0.579\pm0.016$, $-0.0334\pm0.0284$/$-0.0022\pm0.0086$,
frustrated 0.561/0.610) re-spot-checked, unchanged, still exact. Only
change this round: the edge-norm trajectory description corrected to
match the artifact precisely (see finding #1 above) — the figure's
existence, its qualitative shape, and the human-decision caveat on its
interpretation are all unchanged and still stand as written in round 2.

## Remaining open items (by design, not oversight)

1. Four literature-citation markers: `[cite: SocialArcNet]` (×2, §3.1 and
   §4.3), `[cite: DialogueRNN, MMGCN, EmoShiftNet]` (§4.3). Real
   bibliographic entries for external papers — this repository cannot
   supply them; assigned to the human, as instructed.
2. Figure/table cross-reference placeholders (`[EDGE-NORM]`,
   `[K-SWEEP]`) — final numbering depends on paper layout decisions not
   yet made, not a data question.

No other `[CHECK]`, `[INSERT]`, or bracketed placeholder remains in
`docs/SECTION_3_DRAFT.md`, `docs/SECTION_4_DRAFT.md`,
`docs/SECTION_5_DRAFT.md`, or `docs/SECTION_6_DRAFT.md` (grepped all
four files directly to confirm before writing this line).
