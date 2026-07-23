# Table and figure provenance (camera-ready pass, step 8)

The camera-ready copies in `paper/tables/*.tex` and the figure captions in
`paper/sections/*.tex` no longer carry inline `Source: ...` footnotes (those
path/script references read oddly in a published paper). This document is
the redirect target: every table and figure's number, caption, and exact
generating artifacts, so the mapping is not lost, just relocated.

Numbering below matches the compiled PDF (`paper/build/main.pdf`, 13 pages):
IEEEtran numbers tables with roman numerals and figures with arabic numerals,
each in its own sequence, in order of appearance.

Substantive caption remarks that are not themselves path/script references
(e.g. Table V's "$k=0$ anchor is $n=5$, not $n=7$", Table VI's "Pre-registered
hypothesis REFUTED", Table I's "Metric: weighted F1") were **kept** in the
captions — only the path/script provenance was moved here. See
`scripts/regenerate_all_tables.py`'s `booktabs()` helper, which now separates
these two categories explicitly (`source=` vs. `remark=` parameters).

## Tables

### Table I — MELD frozen-feature linear probes
- **Caption**: "MELD frozen-feature linear probes (weighted F1). 'Balanced' trains with `class_weight='balanced'`; both columns report weighted F1 (not a different metric)."
- **Label**: `tab:meld-probe`
- **Source artifact**: `outputs/meld_probe_table.json`
- **Generating script**: `scripts/probe_features.py` (computation), `scripts/regenerate_all_tables.py::table_a_meld_probe` (LaTeX emission)

### Table II — Frozen-regime bridging experiment
- **Caption**: "Frozen-regime bridging experiment: paired per-seed gain, full_R_frozen minus the frozen text anchor, under the identical residual apparatus as the fine-tuned end."
- **Label**: `tab:frozen-bridging`
- **Source artifact**: `outputs/subsumption_curve_data.json['bridging']`
- **Generating script**: `scripts/train_frozen_text_foundation.py`, `scripts/run_bridging_matrix.py` (computation), `scripts/regenerate_all_tables.py::table_b_frozen_bridging` (LaTeX emission)

### Table III — MELD scratch vs. residual attribution
- **Caption**: "MELD, fine-tuned foundation: scratch retraining (N4) vs. residual attribution (N4-R). The full-stack deficit shrinks 63% under residual attribution."
- **Label**: `tab:scratch-vs-residual`
- **Source artifacts**: `outputs/{full,base_fusion}_seed{42,1337,2024}/metrics.json` (scratch), `outputs/{full_R,base_fusion_R}_seed{42,1337,2024}/metrics.json` (residual)
- **Generating script**: `scripts/regenerate_all_tables.py::table_c_scratch_vs_residual`

### Table IV — Per-class test F1, MELD
- **Caption**: "Per-class test F1 (mean of 3 seeds), MELD. Rare classes in **bold**."
- **Label**: `tab:per-class-meld`
- **Source artifacts**: `outputs/{context_text_plain_ce,base_fusion_R,minus_relational_R,full_R}_seed*/metrics.json` (field `test_per_class_f1`)
- **Generating script**: `scripts/regenerate_all_tables.py::table_f_per_class("meld")`

### Table V — Context-window ($k$) sweep, MELD
- **Caption**: "Context-window ($k$) sweep, MELD: paired per-seed gain (full_R − text anchor) at the powered endpoints. $k=0$ text anchor is $n=5$, not $n=7$."
- **Label**: `tab:k-sweep-endpoints`
- **Source artifact**: `outputs/subsumption_curve_data.json['paired_n7_test']`
- **Generating script**: `scripts/regenerate_all_tables.py::table_d_k_sweep_endpoints`
- **Note**: the $n=5$ vs. $n=7$ asymmetry is explained in `docs/PHASE_N5A.md` (the $k=0$ text anchor was seed-42-only in the original design; that consolidation phase added 4 new seeds, not 6).

### Table VI — IEMOCAP pre-registered trial vs. MELD
- **Caption**: "IEMOCAP pre-registered trial (Phase N5-B, Step B5) vs. MELD, matched attribution methodology, $n=3$ seeds each. Pre-registered hypothesis REFUTED."
- **Label**: `tab:iemocap-decisive`
- **Source artifacts**: `outputs/iemocap_text_anchor_seed*`, `outputs/{full_R,minus_relational_R}_iemocap_seed*`, `outputs/context_text_plain_ce_seed*`, `outputs/{full_R,minus_relational_R}_seed*`
- **Generating script**: `scripts/regenerate_all_tables.py::table_e_iemocap_decisive`
- **Note**: the refutation verdict and its decision rule are recorded in `docs/PHASE_N5B.md`.

### Table VII — Per-class test F1, IEMOCAP
- **Caption**: "Per-class test F1 (mean of 3 seeds), IEMOCAP. Pre-registered best-case class in **bold**."
- **Label**: `tab:per-class-iemocap`
- **Source artifacts**: `outputs/{iemocap_text_anchor,base_fusion_R_iemocap,minus_relational_R_iemocap,full_R_iemocap}_seed*/metrics.json` (field `test_per_class_f1`)
- **Generating script**: `scripts/regenerate_all_tables.py::table_f_per_class("iemocap")`

### Table VIII — Per-seed appendix
- **Caption**: "Per-seed appendix: raw test weighted F1 for every run cited in the main tables."
- **Label**: `tab:per-seed-appendix`
- **Source artifact**: every `outputs/*/metrics.json` referenced by Tables III–VI
- **Generating script**: `scripts/regenerate_all_tables.py::table_g_per_seed_appendix`

## Figures

### Figure 1 — The fine-tuning boundary
- **Caption**: "The fine-tuning boundary: graph machinery's paired gain, frozen vs. fine-tuned foundations (MELD). Positive on frozen features; null-to-negative at both fine-tuned endpoints."
- **Label**: `fig:boundary`
- **Source artifact**: `outputs/subsumption_curve_data.json`
- **Generating script**: `scripts/generate_boundary_figure.py`

### Figure 2 — MELD subsumption curve ($k$-sweep)
- **Caption**: "MELD subsumption curve: graph stack's paired residual gain across context-window widths $k \in \{0,2,4,8\}$. Endpoints ($k=0$, $k=8$) powered to $n=5$/$n=7$ seeds with bold error bars; interior points ($k=2,4$) are single-seed trend indicators, de-emphasized."
- **Label**: `fig:k-sweep`
- **Source artifact**: `outputs/subsumption_curve_data.json`
- **Generating script**: `scripts/generate_publication_figures.py`

### Figure 3 — MELD mechanism
- **Caption**: "MELD mechanism: scratch retraining vs. residual attribution. The baseline→full-stack gap shrinks 63% once the incidental-presence confound is removed by construction."
- **Label**: `fig:mechanism`
- **Source artifact**: `outputs/*/metrics.json` (scratch and residual run sets, same as Table III)
- **Generating script**: `scripts/generate_mechanism_figure.py`

### Figure 4 — Relational edge-state trajectory
- **Caption**: "Relational edge-state $L_2$ norm over dialogue time, three complete Session 5 dialogues (`full_R_iemocap`, seed 42)."
- **Label**: `fig:edge-norm`
- **Source artifact**: `outputs/iemocap_edge_state_trajectory_data.json`
- **Generating script**: `scripts/generate_publication_figures.py`

### Figure 5 — Confusion matrices
- **Caption**: "Test-set confusion matrices, row-normalized (recall within true class), matched $[0,1]$ color scale. `full_R` configuration, seed 42, both corpora."
- **Label**: `fig:confusion`
- **Source artifact**: `outputs/confusion_matrices_raw.json`
- **Generating script**: `scripts/generate_confusion_matrices.py` (raw JSON; **not** re-runnable from a fresh clone without the large `best_model.pt` checkpoints for `full_R_seed42` and `full_R_iemocap_seed42` — the figure itself IS reproducible from the archived JSON alone, per `docs/REPRODUCE.md`)

## Regeneration

```
uv run python -m scripts.regenerate_all_tables          # -> paper_assets/tables/*.tex, full Source footnotes (reproducibility archive)
uv run python -m scripts.regenerate_all_tables --paper  # -> paper/tables/*.tex, Source footnotes stripped (camera-ready copy, this document is the provenance record instead)
```

Both modes read the same `outputs/*.json` artifacts and differ only in
whether the `Source: ...` footnote is emitted; no number differs between
the two modes.
