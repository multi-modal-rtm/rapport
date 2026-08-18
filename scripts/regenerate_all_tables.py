"""PAPER-ASSETS PHASE: single-source regeneration of every paper table.

Reads ONLY stored artifacts (outputs/*/metrics.json, outputs/*.json) --
no number in this script's OUTPUT tables is hand-typed. Every table is a
pure function of files already committed/present under outputs/.

Default mode emits LaTeX booktabs tables to paper_assets/tables/*.tex,
each with a full "Source: ..." provenance footnote -- this is the
reproducibility-archive copy (see docs/REPRODUCE.md, which diffs it
byte-for-byte against a fresh clone's regeneration):
  (a) meld_probe_table.tex         -- per-modality + concat probes
  (b) frozen_bridging_table.tex    -- frozen-regime bridging experiment
  (c) meld_scratch_vs_residual.tex -- N4 (scratch) vs N4-R (residual)
  (d) k_sweep_endpoints.tex        -- n=7/n=5 endpoint paired gains
  (e) iemocap_decisive_table.tex   -- IEMOCAP anchor + B3 vs MELD columns
  (f) per_class_meld.tex, per_class_iemocap.tex
  (g) per_seed_appendix.tex

`--paper` mode emits the same tables to paper/tables/*.tex instead, with
the path/script "Source: ..." footnote stripped (that provenance lives in
docs/TABLE_PROVENANCE.md for camera-ready copy); substantive remarks
that are not themselves path/script references (e.g. "Metric: weighted
F1", "Pre-registered hypothesis REFUTED") are kept in both modes.

Usage:
    uv run python -m scripts.regenerate_all_tables
    uv run python -m scripts.regenerate_all_tables --paper
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = PROJECT_ROOT / "paper_assets" / "tables"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "tables"
PAPER_MODE = "--paper" in sys.argv[1:]

MELD_SEEDS = (42, 1337, 2024)
IEMOCAP_SEEDS = (42, 1337, 2024)


def tex_escape(s: str) -> str:
    """Escapes underscores for safe use outside LaTeX math mode (run names/
    JSON field names contain literal underscores that break compilation
    otherwise -- caught when the tectonic-compiled paper first surfaced
    unescaped underscores in these two footnotes)."""
    return s.replace("_", "\\_")


def display_name(raw: str) -> str:
    """Maps a raw Hydra config key to the reader-facing label used
    consistently across every table and figure -- no raw key (full_R,
    base_fusion_R, minus_relational_R, context_text_plain_ce,
    iemocap_text_anchor, or any _iemocap/_frozen-suffixed variant of
    these) may appear where a reader can see it."""
    if raw in ("context_text_plain_ce", "iemocap_text_anchor", "text anchor"):
        return "Text-only anchor"
    if raw.startswith("minus_relational_R"):
        return "Full stack (no relational)" + (" (frozen)" if raw.endswith("_frozen") else "")
    if raw.startswith("base_fusion_R"):
        return "Fusion" + (" (frozen)" if raw.endswith("_frozen") else "")
    if raw.startswith("full_R"):
        return "Full stack" + (" (frozen)" if raw.endswith("_frozen") else "")
    return raw


def load(run_name: str) -> dict:
    return json.loads((OUTPUTS_DIR / run_name / "metrics.json").read_text())


def load_json(name: str) -> dict:
    return json.loads((OUTPUTS_DIR / name).read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def wf1(run_name: str) -> float:
    return load(run_name)["test_weighted_f1"]


def booktabs(
    caption: str,
    label: str,
    col_spec: str,
    header: list[str],
    rows: list[list[str]],
    source: str = "",
    remark: str = "",
    wide: bool = False,
) -> str:
    """`wide=True` emits a `table*` (spans both columns of a two-column
    IEEEtran layout) instead of a single-column `table` -- needed for
    tables whose header/cell content (e.g. full run-config names as
    column headers) overflows a ~3.5in single column. Set per-table by
    the caller based on measured column content, not guessed in advance.

    `source` is path/script provenance ("Source: outputs/....json
    (scripts/....py)."); it is dropped in `--paper` mode, where that
    mapping instead lives in docs/TABLE_PROVENANCE.md. `remark` is a
    substantive caption note that isn't a path/script reference (e.g.
    a metric clarification or a result annotation); it is kept in both
    modes."""
    env = "table*" if wide else "table"
    lines = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    notes_parts = [p for p in (remark, "" if PAPER_MODE else source) if p]
    if notes_parts:
        lines.append(rf"\par\footnotesize {' '.join(notes_parts)}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def write(name: str, content: str) -> None:
    out_dir = PAPER_TABLES_DIR if PAPER_MODE else TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(content)
    print(f"[regenerate_all_tables] wrote {path}")


# ---------------------------------------------------------------- (a) ----
def table_a_meld_probe() -> None:
    data = load_json("meld_probe_table.json")
    rows_by_name = {r["name"]: r for r in data["rows"]}
    order = ["all-neutral (constant)", "random-stratified", "video", "audio", "text", "concat"]
    rows = []
    for name in order:
        r = rows_by_name[name]
        rows.append([name, f"{r['unbalanced_weighted_f1']:.4f}", f"{r['balanced_weighted_f1']:.4f}"])
    content = booktabs(
        caption="MELD frozen-feature linear probes (weighted F1). ``Balanced'' trains with "
        "\\texttt{class\\_weight=`balanced'}; both columns report weighted F1 (not a different metric).",
        label="tab:meld-probe",
        col_spec="lcc",
        header=["row", "unbalanced", "balanced"],
        rows=rows,
        source="Source: outputs/meld\\_probe\\_table.json (scripts/probe\\_features.py).",
        remark=f"Metric: {tex_escape(data['metric'])}.",
    )
    write("meld_probe_table.tex", content)


# ---------------------------------------------------------------- (b) ----
def table_b_frozen_bridging() -> None:
    data = load_json("subsumption_curve_data.json")["bridging"]
    anchor = data["frozen_anchor_test_weighted_f1"]
    base = data["base_fusion_R_frozen"]
    full = data["full_R_frozen"]
    pm, ps = data["paired_mean"], data["paired_std"]

    rows = [
        ["Text-only anchor (frozen, seed 42)", f"{anchor:.4f}", "--", "1"],
        [display_name("base_fusion_R_frozen"), f"{base['mean']:.4f} $\\pm$ {base['std']:.4f}", "--", str(base["n_seeds"])],
        [display_name("full_R_frozen"), f"{full['mean']:.4f} $\\pm$ {full['std']:.4f}", f"{pm:+.4f} $\\pm$ {ps:.4f}", str(full["n_seeds"])],
    ]
    content = booktabs(
        caption="Frozen-regime bridging experiment: paired per-seed gain, Full stack (frozen) minus the "
        "frozen Text-only anchor, under the identical residual apparatus used for the fine-tuned foundation.",
        label="tab:frozen-bridging",
        col_spec="lccc",
        header=["config", "test weighted F1", "paired gain vs.\\ anchor", "n seeds"],
        rows=rows,
        source="Source: outputs/subsumption\\_curve\\_data.json['bridging'] "
        "(scripts/train\\_frozen\\_text\\_foundation.py, scripts/run\\_bridging\\_matrix.py).",
        wide=True,  # config-name column ("base_fusion_R_frozen") overflows a single IEEEtran column
    )
    write("frozen_bridging_table.tex", content)


# ---------------------------------------------------------------- (c) ----
def table_c_scratch_vs_residual() -> None:
    scratch_full = [wf1(f"full_seed{s}") for s in MELD_SEEDS]
    scratch_base = [wf1(f"base_fusion_seed{s}") for s in MELD_SEEDS]
    residual_full = [wf1(f"full_R_seed{s}") for s in MELD_SEEDS]
    residual_base = [wf1(f"base_fusion_R_seed{s}") for s in MELD_SEEDS]

    scratch_gap = statistics.mean(scratch_full) - statistics.mean(scratch_base)
    residual_gap = statistics.mean(residual_full) - statistics.mean(residual_base)
    shrinkage_pct = (1 - abs(residual_gap) / abs(scratch_gap)) * 100 if scratch_gap != 0 else float("nan")

    sf_m, sf_s = mean_std(scratch_full)
    sb_m, sb_s = mean_std(scratch_base)
    rf_m, rf_s = mean_std(residual_full)
    rb_m, rb_s = mean_std(residual_base)

    rows = [
        ["Fusion", f"{sb_m:.4f} $\\pm$ {sb_s:.4f}", f"{rb_m:.4f} $\\pm$ {rb_s:.4f}"],
        ["Full stack", f"{sf_m:.4f} $\\pm$ {sf_s:.4f}", f"{rf_m:.4f} $\\pm$ {rf_s:.4f}"],
        ["gap (Full stack $-$ Fusion)", f"{scratch_gap:+.4f}", f"{residual_gap:+.4f}"],
    ]
    content = booktabs(
        caption=f"MELD, fine-tuned foundation: scratch retraining vs.\\ residual attribution. "
        f"The full-stack deficit shrinks {shrinkage_pct:.0f}\\% under residual attribution.",
        label="tab:scratch-vs-residual",
        col_spec="lcc",
        header=["", "scratch retraining", "residual attribution"],
        rows=rows,
        source="Source: outputs/\\{full,base\\_fusion\\}\\_seed\\{42,1337,2024\\}/metrics.json (scratch), "
        "outputs/\\{full\\_R,base\\_fusion\\_R\\}\\_seed\\{42,1337,2024\\}/metrics.json (residual).",
        wide=True,  # "gap (Full stack $-$ Fusion)" row label overflows a single IEEEtran column
    )
    write("meld_scratch_vs_residual.tex", content)


# ---------------------------------------------------------------- (d) ----
def table_d_k_sweep_endpoints() -> None:
    data = load_json("subsumption_curve_data.json")
    paired = data["paired_n7_test"]

    rows = []
    for k in ("0", "8"):
        p = paired[k]
        rows.append([f"$k={k}$", str(p["n_pairs"]), f"{p['mean']:+.4f} $\\pm$ {p['std']:.4f}", str(abs(p["mean"]) > p["std"])])
    content = booktabs(
        caption="Context-window ($k$) sweep, MELD: paired per-seed gain (Full stack $-$ Text-only anchor) "
        "at the powered endpoints.",
        label="tab:k-sweep-endpoints",
        col_spec="lccc",
        header=["endpoint", "$n$ pairs", "paired gain", "$|$gain$|>$std"],
        rows=rows,
        source="Source: outputs/subsumption\\_curve\\_data.json['paired\\_n7\\_test'] (docs/PHASE\\_N5A.md).",
        remark="$k=0$ text anchor is $n{=}5$, not $n{=}7$.",
    )
    write("k_sweep_endpoints.tex", content)


# ---------------------------------------------------------------- (e) ----
def table_e_iemocap_decisive() -> None:
    iemocap_anchor = {s: wf1(f"iemocap_text_anchor_seed{s}") for s in IEMOCAP_SEEDS}
    iemocap_full = {s: wf1(f"full_R_iemocap_seed{s}") for s in IEMOCAP_SEEDS}
    iemocap_minus_rel = {s: wf1(f"minus_relational_R_iemocap_seed{s}") for s in IEMOCAP_SEEDS}

    meld_anchor = {s: wf1(f"context_text_plain_ce_seed{s}") for s in MELD_SEEDS}
    meld_full = {s: wf1(f"full_R_seed{s}") for s in MELD_SEEDS}
    meld_minus_rel = {s: wf1(f"minus_relational_R_seed{s}") for s in MELD_SEEDS}

    def gains(anchor, full, minus_rel, seeds):
        g1 = {s: full[s] - anchor[s] for s in seeds}
        g2 = {s: full[s] - minus_rel[s] for s in seeds}
        return mean_std(list(g1.values())), mean_std(list(g2.values()))

    (ig1m, ig1s), (ig2m, ig2s) = gains(iemocap_anchor, iemocap_full, iemocap_minus_rel, IEMOCAP_SEEDS)
    (mg1m, mg1s), (mg2m, mg2s) = gains(meld_anchor, meld_full, meld_minus_rel, MELD_SEEDS)

    rows = [
        ["Full stack $-$ Text-only anchor", f"{ig1m:+.4f} $\\pm$ {ig1s:.4f}", f"{mg1m:+.4f} $\\pm$ {mg1s:.4f}"],
        ["Full stack $-$ Full stack (no relational)", f"{ig2m:+.4f} $\\pm$ {ig2s:.4f}", f"{mg2m:+.4f} $\\pm$ {mg2s:.4f}"],
    ]
    content = booktabs(
        caption="IEMOCAP pre-registered trial vs.\\ MELD, matched attribution "
        "methodology, $n{=}3$ seeds each.",
        label="tab:iemocap-decisive",
        col_spec="lcc",
        header=["paired gain", "IEMOCAP", "MELD"],
        rows=rows,
        source="Source: outputs/iemocap\\_text\\_anchor\\_seed*, outputs/\\{full\\_R,minus\\_relational\\_R\\}\\_iemocap\\_seed*, "
        "outputs/context\\_text\\_plain\\_ce\\_seed*, outputs/\\{full\\_R,minus\\_relational\\_R\\}\\_seed* (docs/PHASE\\_N5B.md).",
        remark="Pre-registered hypothesis REFUTED.",
        wide=True,  # long source footnote overflows a single IEEEtran column
    )
    write("iemocap_decisive_table.tex", content)


# ---------------------------------------------------------------- (f) ----
def table_f_per_class(corpus: str) -> None:
    if corpus == "meld":
        configs = ["context_text_plain_ce", "base_fusion_R", "minus_relational_R", "full_R"]
        seeds = MELD_SEEDS
        highlight = {"fear", "disgust"}
        label_suffix = "meld"
    else:
        configs = ["iemocap_text_anchor", "base_fusion_R_iemocap", "minus_relational_R_iemocap", "full_R_iemocap"]
        seeds = IEMOCAP_SEEDS
        highlight = {"frustrated"}
        label_suffix = "iemocap"

    per_config_mean: dict[str, dict[str, float]] = {}
    class_order: list[str] = []
    for cfg in configs:
        dicts = [load(f"{cfg}_seed{s}")["test_per_class_f1"] for s in seeds]
        keys = list(dicts[0].keys())
        if not class_order:
            class_order = keys
        per_config_mean[cfg] = {k: statistics.mean(d[k] for d in dicts) for k in keys}

    rows = []
    for cls in class_order:
        cls_label = f"\\textbf{{{cls}}}" if cls in highlight else cls
        row = [cls_label] + [f"{per_config_mean[cfg][cls]:.3f}" for cfg in configs]
        rows.append(row)

    header = ["class"] + [display_name(c) for c in configs]
    content = booktabs(
        caption=f"Per-class test F1 (mean of {len(seeds)} seeds), {corpus.upper()}. "
        f"{'Rare classes' if corpus == 'meld' else 'Pre-registered best-case class'} in \\textbf{{bold}}.",
        label=f"tab:per-class-{label_suffix}",
        col_spec="l" + "c" * len(configs),
        header=header,
        rows=rows,
        source=f"Source: outputs/\\{{{','.join(tex_escape(c) for c in configs)}\\}}\\_seed*/metrics.json ('test\\_per\\_class\\_f1').",
        wide=True,  # 4 full run-config names as column headers overflow a single IEEEtran column
    )
    write(f"per_class_{label_suffix}.tex", content)


# ---------------------------------------------------------------- (g) ----
def table_g_per_seed_appendix() -> None:
    # Display names match the convention used in every other table
    # (full_R (k=8) -> "Full stack (k=8)"; plain full_R -> "Full stack",
    # since IEMOCAP has no k-sweep to disambiguate; minus_relational_R ->
    # "Full stack (no relational)"; base_fusion_R -> "Fusion"; text
    # anchor -> "Text-only anchor"), each still corpus-prefixed.
    meld_groups = [
        ("MELD Text-only anchor", [f"context_text_plain_ce_seed{s}" for s in MELD_SEEDS]),
        ("MELD Full stack (k=8)", [f"full_R_seed{s}" for s in MELD_SEEDS]),
        ("MELD Full stack (no relational)", [f"minus_relational_R_seed{s}" for s in MELD_SEEDS]),
        ("MELD Fusion", [f"base_fusion_R_seed{s}" for s in MELD_SEEDS]),
    ]
    iemocap_groups = [
        ("IEMOCAP Text-only anchor", [f"iemocap_text_anchor_seed{s}" for s in IEMOCAP_SEEDS]),
        ("IEMOCAP Full stack", [f"full_R_iemocap_seed{s}" for s in IEMOCAP_SEEDS]),
        ("IEMOCAP Full stack (no relational)", [f"minus_relational_R_iemocap_seed{s}" for s in IEMOCAP_SEEDS]),
        ("IEMOCAP Fusion", [f"base_fusion_R_iemocap_seed{s}" for s in IEMOCAP_SEEDS]),
    ]

    def block_rows(groups: list[tuple[str, list[str]]]) -> list[str]:
        lines = [r"\toprule", r"config & seed & test weighted F1 \\", r"\midrule"]
        for gi, (label, run_names) in enumerate(groups):
            if gi > 0:
                lines.append(r"\addlinespace")
            for run_name in run_names:
                seed = load(run_name)["seed"]
                lines.append(rf"{label} & {seed} & {wf1(run_name):.4f} \\")
                label = ""  # only print the group label once
        lines.append(r"\bottomrule")
        return lines

    # \small, a tighter \tabcolsep, and a fixed-width first column: the
    # corpus-prefixed display labels ("IEMOCAP Full stack (no relational)")
    # are wider than the raw config-key labels this table used to show. A
    # plain "l" column never wraps, so it overflowed the 0.48\textwidth
    # minipage; p{2.6cm} fixed that but wrapped even short labels onto
    # three lines, ballooning the table's height enough to push the
    # references onto a new page. p{3.5cm} keeps every label at one line
    # except the single longest ("...(no relational)"), which wraps to
    # two -- back to roughly the original row height.
    col_spec = r"p{3.5cm}lc"
    preamble = [r"\small", r"\setlength{\tabcolsep}{3.5pt}"]
    meld_tabular = "\n".join(preamble + [rf"\begin{{tabular}}{{{col_spec}}}"] + block_rows(meld_groups) + [r"\end{tabular}"])
    iemocap_tabular = "\n".join(preamble + [rf"\begin{{tabular}}{{{col_spec}}}"] + block_rows(iemocap_groups) + [r"\end{tabular}"])

    source = "" if PAPER_MODE else "Source: every outputs/*/metrics.json referenced by tables (c)-(e)."
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-seed appendix: raw test weighted F1 for every run cited in the main tables.}",
        r"\label{tab:per-seed-appendix}",
        r"\begin{minipage}{0.48\textwidth}",
        r"\centering",
        meld_tabular,
        r"\end{minipage}%",
        r"\hfill",
        r"\begin{minipage}{0.48\textwidth}",
        r"\centering",
        iemocap_tabular,
        r"\end{minipage}",
    ]
    if source:
        lines.append(rf"\par\footnotesize {source}")
    lines.append(r"\end{table*}")
    write("per_seed_appendix.tex", "\n".join(lines) + "\n")


def main() -> None:
    table_a_meld_probe()
    table_b_frozen_bridging()
    table_c_scratch_vs_residual()
    table_d_k_sweep_endpoints()
    table_e_iemocap_decisive()
    table_f_per_class("meld")
    table_f_per_class("iemocap")
    table_g_per_seed_appendix()
    print("[regenerate_all_tables] done")


if __name__ == "__main__":
    main()
