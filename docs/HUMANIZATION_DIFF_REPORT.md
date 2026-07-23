# Humanization diff report (CAMERA-READY PASS, step 3 + step 6 verification)

Full diff of `paper/sections/*.tex` against commit `4eb386e` (the prior
LaTeX-port commit), reviewed sentence by sentence. This report separates
three categories of change, per the task's own request ("listing every
sentence whose words, not just punctuation, changed"):

1. **Pure punctuation substitutions** (the overwhelming majority) — dash
   removed, comma/semicolon/colon/parentheses inserted, zero words added,
   removed, or reordered.
2. **Word-level changes from punctuation restructuring** (3 sentences,
   listed in full below) — restructuring an em-dash construction
   sometimes requires a small connective word (``and'', ``including'',
   a repeated subject for a sentence split); these are flagged
   individually rather than folded into category 1.
3. **Citation-key changes** (step 4, bibliography rebuild — not step 3
   humanization, listed separately for transparency since they also
   touch `\cite{}` text inside sentences).

## Em-dash count

| file | before | after |
|---|---|---|
| `sections/1_introduction.tex` | 9 | 0 |
| `sections/2_related_work.tex` | 10 | 0 |
| `sections/3_methodology.tex` | 18 | 0 |
| `sections/4_meld_experiments.tex` | 25 | 0 |
| `sections/5_iemocap.tex` | 13 | **1** (protected, see below) |
| `sections/6_analysis.tex` | 21 | 0 |
| `sections/7_conclusion.tex` | 4 | 0 |
| `sections/8_appendix.tex` | 0 | 0 |
| **Total** | **100** | **1** |

**The one retained dash** (`sections/5_iemocap.tex`, inside
`\begin{quotation}...\end{quotation}`): *"Either outcome --- confirmed or
not --- gets reported..."* — this is inside the verbatim pre-registered
hypothesis quote. Per the hard constraint ("the verbatim hypothesis quote
in Section V keeps its original punctuation exactly"), it was **not
touched**, by design, not oversight. All other 99 constructions were
restructured (not globally substituted) per instance: parenthetical
asides to parentheses, contrastive/appositive dashes to commas,
semicolons, or colons per grammar, and one long dash-spliced sentence
(Section 4.3) split into two.

## Category 2: sentences where words (not just punctuation) changed

Exactly three, all minor connective insertions/deletions required by the
surrounding restructuring, none touching a numeric value, table, caption,
citation, or the protected quote:

**1. `sections/1_introduction.tex`, opening paragraph.**
- Before: "...propagate influence between them --- DialogueGCN~\cite{...}, MMGCN~\cite{...}, COGMEN~\cite{...}, and our own prior SocialArcNet~\cite{...} (published: $0.62$ weighted F1 on MELD) among them."
- After: "...propagate influence between them, including DialogueGCN~\cite{...}, MMGCN~\cite{...}, COGMEN~\cite{...}, and our own prior SocialArcNet~\cite{...} (published: $0.62$ weighted F1 on MELD)."
- Change: dropped the trailing "among them" (redundant with the new "including"), added "including" to introduce the list without a dash. No citation, number, or claim changed.

**2. `sections/3_methodology.tex`, §III-C (Components Under Test), shift-objective premise sentence.**
- Before: "...on IEMOCAP the same check is only partially confirmed --- Section~\ref{sec:dissociation} reports both in full)."
- After: "...on IEMOCAP the same check is only partially confirmed, and Section~\ref{sec:dissociation} reports both in full)."
- Change: added "and" to join the two clauses with a comma-conjunction instead of a dash.

**3. `sections/4_meld_experiments.tex`, §IV-C (The Fine-Tuned Regime), boundary-claim summary sentence.**
- Before: "...the same recipe, the same seeds --- a $+0.036$ contribution on frozen features and a null-to-negative contribution on fine-tuned ones."
- After: "...the same recipe, the same seeds. The result is a $+0.036$ contribution on frozen features and a null-to-negative contribution on fine-tuned ones."
- Change: split into two sentences (this was the one case matching the instruction's own "long dash-spliced sentences -> split into two" rule); added "The result is" as the new sentence's subject and verb. The $+0.036$ figure itself is unchanged.

No other sentence in any of the seven files had a word added, removed, or
reordered by the humanization pass — every other em-dash construction was
resolved by punctuation alone (dash to comma / semicolon / colon /
parentheses). This was cross-checked with an independent, automated
word-token diff (Python, LaTeX-command-stripped word-multiset comparison)
against the pre-pass commit, not just manual review: the tool confirms
exactly these three sentences and no others changed at the word level in
`sections/1_introduction.tex`, `3_methodology.tex`, and
`4_meld_experiments.tex`; `sections/5_iemocap.tex`, `6_analysis.tex`, and
`7_conclusion.tex` show a **zero-word-diff** result, i.e. every edit in
those three files was verified to be pure punctuation with no textual
side effects at all.

**One caveat surfaced by the automated check, noted for completeness**:
the `%`-prefixed LaTeX source comments at the top of
`sections/2_related_work.tex` and `sections/4_meld_experiments.tex` were
also rewritten in this pass (updating stale references to now-resolved
`TODO-` citation markers from the prior LaTeX-port commit). These are
**not part of the paper's rendered prose** (invisible in the compiled
PDF) and are unrelated to punctuation humanization; the word-token tool
flags them because it diffs the whole file, not just visible text. Listed
here rather than silently excluded, since the task asked for every word
change to be reported.

## Category 3: citation-key changes (step 4, not step 3 — listed for transparency)

These touch `\cite{}` calls' key text but are bibliography-rebuild work
(task step 4), not punctuation humanization (step 3); listed here so the
full diff is fully accounted for, not because they violate step 3's "no
citation may change" constraint (that constraint governs step 3's own
restructuring, not step 4's separate, explicitly authorized rebuild):

- `mmgcn2019` -> `mmgcn2021` (3 occurrences: sections 1, 2, 4) — resolved
  MMGCN conflict in favor of the correct ERC paper.
- `spcl2021` -> `spcl2022` (section 2) — corrected wrong-paper SPCL entry.
- `TODO-peft-affective-computing-representative-works` -> `feng2023peftser`
- `TODO-unequal-optimization-budgets` -> `missbench2026,lucic2018gans,melis2017evaluation`
- `TODO-preregistration-negative-results` -> `farhadipour2025multimodal,pineau2021reproducibility`
- `TODO-zero-init-residual-precedent` -> `bachlechner2020rezero,zhang2019fixup`
- `TODO-graph-less-node-classification` -> `zhang2021graphless,huang2020labelprop`
- `TODO-poria2019-meld` -> `poria2019meld`

Full detail (conflict resolution reasoning, unused-entry disposition) is
in `paper/references.bib`'s own header comments and the chat report for
this pass.

## Verification performed

- **Number audit re-run** (this report's step 6 requirement): every
  3-4-decimal-place result number in `paper/sections/*.tex` matches its
  `docs/SECTION_*_DRAFT.md` source exactly, before and after this pass
  (strict regex diff, zero mismatches in either direction).
- **Recompiled clean**: `tectonic`, zero errors, zero Overfull/Underfull
  warnings introduced by this pass, zero BibTeX warnings, 13 pages
  (unchanged from the prior LaTeX-port commit).
- **Tables, captions, figures**: not touched by this pass at all (step 3
  scoped changes to `sections/*.tex` prose only; `paper/tables/*.tex` and
  figure captions were not edited).
