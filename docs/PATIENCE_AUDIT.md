# Patience audit — MELD vs. IEMOCAP residual matrices

## Pre-committed remedy (recorded BEFORE the investigation below; nothing
past this section has been written yet as of this commit)

If finding 3 fires (patience differs across corpora AND `docs/RECIPE.md`
is silent or contradicted on the difference), the remedy is a
**harmonized-patience re-run of the IEMOCAP B3 matrix, reported as an
APPENDIX robustness check only.** The title decision (branch a, applied in
commit `38eaab2`) is **not re-litigated** by it. If the robustness run
disagrees with the original result, both results are reported with the
disagreement stated plainly — the original B3 result is not silently
superseded or discarded.

**No re-runs are authorized by the audit that follows this section.**
Everything below is read-only investigation of existing `resolved_config.yaml`
snapshots and `docs/RECIPE.md`.

## Method note: `resolved_config.yaml` does not exist for these runs

Checked first, before anything else: no run directory under `outputs/`
(MELD or IEMOCAP) contains a `resolved_config.yaml`. `src/rapport/repro.py`'s
`snapshot_environment(run_dir, cfg=None)` only writes that file when a Hydra
`cfg` is passed in; `scripts/train_rapport.py` and
`scripts/train_rapport_iemocap.py` are both standalone scripts (not part of
the Hydra `rapport.__main__` pipeline) and call `snapshot_environment(run_dir)`
with no `cfg` at all — so the file is never written for any run in either
matrix. This is not new to IEMOCAP; it's true of every `train_rapport*.py`
run in the project's history.

**What each run dir has instead**: `git_commit.txt`, the exact repo commit
the run executed under. Since `EARLY_STOP_PATIENCE`/`MAX_EPOCHS`/the
selection-metric line are hardcoded Python module constants in
`scripts/train_rapport.py` / `scripts/train_rapport_iemocap.py` (not
config-driven), `git show <commit>:scripts/train_rapport*.py` recovers the
EXACT values in effect for that run precisely — arguably more reliable
than a resolved-config snapshot would have been, since it's tied to a
content-addressed commit rather than a value that could in principle be
edited without a corresponding commit.

## Investigation

Every relevant run's `git_commit.txt`, and the constants in effect at each commit:

| context | representative runs | commit | `EARLY_STOP_PATIENCE` | `MAX_EPOCHS` | selection metric |
|---|---|---|---|---|---|
| MELD, N4-R Step 4 | `full_R_seed{42,1337,2024}`, `minus_relational_R_seed{42,1337,2024}` | `8efca4a` | 10 | 100 | val macro F1 |
| MELD, N5-A endpoint (original 3-seed, k=0) | `full_R_k0_seed42`, `base_fusion_R_k0_seed42` | `95733ae` | 10 | 100 | val macro F1 |
| MELD, N5-A endpoint (n=7 power-up seeds) | `full_R_seed{7,123,555,9090}`, `full_R_k0_seed{7,9090}` | `04d9267` | 10 | 100 | val macro F1 |
| IEMOCAP, B3 matrix | `{base_fusion_R,full_R,minus_relational_R}_iemocap_seed{42,1337,2024}` | `2ea0201` (HEAD when the run started; `scripts/train_rapport_iemocap.py` itself was uncommitted at run time, committed unchanged at `38eaab2`) | 10 | 100 | val macro F1 |

`scripts/train_rapport.py` was checked at all three commits it was ever
run under across this project's full history (`8efca4a`, `95733ae`,
`04d9267`) — the constants never changed; the file has never been edited
since it was written. `scripts/train_rapport_iemocap.py` (checked as
committed at `38eaab2`, identical to what actually ran) is a direct,
line-for-line derivative of `train_rapport.py` for exactly these three
constants (see `docs/PHASE_N5B.md`'s B5.1: "identical apparatus to MELD").

**All four values are identical across every context checked: patience
10, max epochs 100, selection metric val macro F1.**

## `docs/RECIPE.md`, quoted for each context

**The locked GNN recipe table** (`docs/RECIPE.md`, "Locked recipe (final)"):
> | max epochs | 100 | original |
> | early stop | patience 10, **metric: val weighted F1** | original (audited twice, not changed — see below) |

Patience (10) and max epochs (100) match what actually ran, exactly, in
both corpora — no drift on either value.

**The selection-metric line** says "val weighted F1", but every run in
both matrices actually selects/early-stops on **val macro F1**. This is
NOT an unexplained contradiction: it is Phase N4's own documented,
pre-existing exception, stated explicitly in `docs/PHASE_N4.md`:
> checkpoint selection / early stopping use **val macro F1**, not
> RECIPE.md's original val weighted F1 convention.

and restated in `scripts/train_rapport.py`'s own header comment ("loss and
the selection metric (val macro F1, not weighted F1) are Phase N4's
documented exceptions"). This exception predates IEMOCAP entirely (it was
adopted for Phase N4, MELD-only, long before Phase N5-B existed) and
`scripts/train_rapport_iemocap.py` inherits it unchanged, symmetrically,
same as every constant checked above — it is not something introduced
newly, or asymmetrically, for IEMOCAP.

*(For contrast, not directly in scope here: the separate Phase T text-
encoder recipe, patience 3 / max 10 epochs / val macro F1, governs
`train_context_text.py` / `train_context_text_iemocap.py` -- the text-
anchor scripts, not the residual-matrix scripts this audit covers. Also
identical across both corpora, checked as a side note, not the audit's
subject.)*

## Finding: **1 — the flag is moot**

Both matrices used the same patience (10), the same max epochs (100), and
the same selection metric (val macro F1, a documented Phase N4 exception
applied symmetrically to both corpora). `docs/RECIPE.md` correctly
specifies patience and max-epochs for both; the selection-metric line's
literal text ("val weighted F1") is superseded by a named, pre-existing,
already-documented exception that both corpora inherit identically, not a
per-corpus discrepancy.

**No footnote is added to `docs/PHASE_N5B.md`'s cross-corpus table** (finding
2 does not apply — there is no difference to document). **The
pre-committed remedy above does not trigger** (finding 3 does not apply —
`RECIPE.md` is neither silent nor contradicted; it is fully consistent
with a named exception both corpora follow the same way). The B3 title
decision (branch a, commit `38eaab2`) stands, unaffected and
unre-litigated, exactly as it would have regardless of this audit's
outcome per remedy finding 1/2's framing.

Closed.
