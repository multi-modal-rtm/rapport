# Reproducing the paper's tables and figures

## What "reproduce" means here — three tiers, be precise about which one you want

1. **Tier 1 — regenerate tables/figures from the stored experimental
   record.** Fast (seconds), no GPU, no data download. This is what
   `scripts/regenerate_all_tables.py` and the `scripts/generate_*.py`
   figure scripts do. **This is the tier `paper_assets/` in this repo was
   built from, and the only tier verified end-to-end below.**
2. **Tier 2 — re-run inference on already-trained checkpoints** (e.g.
   `scripts/generate_confusion_matrices.py`'s live forward pass,
   `scripts/verify_text_ctx_cache*.py`'s process-independence checks).
   Needs a GPU and the `outputs/*/best_model.pt` checkpoint files, which
   are **not** committed to this repo (see "What's NOT included" below).
3. **Tier 3 — reproduce the experimental campaign from raw data.** Full
   preprocessing + feature caching + training, MELD and IEMOCAP, every
   phase documented in `docs/PHASE_*.md`. Multi-day GPU effort (see wall-
   clock notes below); needs both datasets acquired under their own
   licenses (see "Dataset acquisition" below). **Not exercised by this
   doc's smoke test** — this doc points you at the phase docs instead of
   duplicating them.

## Tier 1: clean clone → all tables (verified)

```bash
git clone <repo-url> rapport && cd rapport
uv sync                                    # or: pip install -e .
cp -r paper_assets/results_archive/* outputs/ 2>/dev/null || \
  mkdir -p outputs && cp -r paper_assets/results_archive/* outputs/
uv run pytest tests/ -q --ignore=tests/test_probe_floor.py
uv run python -m scripts.regenerate_all_tables
uv run python -m scripts.generate_publication_figures   # k-sweep + edge-state trajectory
uv run python -m scripts.generate_boundary_figure
uv run python -m scripts.generate_mechanism_figure
```

**Why the copy step is necessary, stated plainly**: `outputs/` and
`data/` are both repo-gitignored (`.gitignore`'s first two lines) — a
literal `git clone` has neither. `paper_assets/results_archive/` is a
**lightweight, committed mirror of ONLY the `metrics.json` files
(and the top-level derived JSONs `subsumption_curve_data.json`,
`meld_probe_table.json`, `iemocap_edge_state_trajectory_data.json`,
`confusion_matrices_raw.json`) that `scripts/regenerate_all_tables.py`
and the figure scripts actually read** — 52 run directories, 1.1MB total,
no model weights, no cached features, no raw data. Copying it into
`outputs/` before running the scripts is the one manual step a fresh
clone needs; everything downstream of that copy is a pure function of
committed, versioned JSON.

`scripts/generate_confusion_matrices.py` (figure (e)'s underlying
computation) is **not** re-runnable from a fresh clone — it needs
`outputs/full_R_seed42/best_model.pt` and
`outputs/full_R_iemocap_seed42/best_model.pt` (Tier 2), which are large
binary checkpoints not included in the lightweight archive. The FIGURE
itself, however, IS reproducible from the archived
`confusion_matrices_raw.json` alone; a from-scratch re-derivation of that
JSON (re-running live inference) needs the checkpoints.

### Fresh-clone smoke test performed for this doc

Executed exactly the Tier 1 sequence above against a genuine
`git clone` of this repository into a scratch directory (not the working
tree this doc was written in), **after** committing this phase's changes,
on this machine. Result: `uv sync` succeeded; `pytest` reported **83
passed, 15 skipped, 1 xfailed** — fewer passes than the working tree's 98,
because 15 tests are correctly guarded to skip when `data/` (the full
MELD/IEMOCAP feature caches, not archived — see "What's NOT included"
below) is absent, which is the honest, expected result for a
`metrics.json`-only archive, not a failure; `regenerate_all_tables.py`
produced all 8 `.tex` files byte-identical to the working tree's
`paper_assets/tables/` (`diff -rq`, zero output); all four
non-checkpoint-dependent figure-generation scripts ran without error and
reproduced their outputs.

## Hardware / wall-clock notes (Tier 3, for context — not reproduced here)

From the phase docs' own logged `avg_epoch_time_sec`/`wall_clock_sec`
fields (`outputs/*/metrics.json`), on the single-GPU machine this project
ran on throughout:

| campaign | approx. wall clock |
|---|---|
| MELD feature caching (V/A/T, full corpus) | tens of minutes (`docs/build_feature_cache*.log`) |
| Phase T / IEMOCAP text-anchor training, 1 seed | ~100–150s (LoRA RoBERTa, few epochs, early-stop patience 3) |
| RAPPORT GNN training (`train_rapport*.py`), 1 run | ~5–20 min depending on config (relational/shift/temporal add cost) and how early it stops (patience 10, up to 100 epochs) |
| Full N4 ablation matrix (15 runs) | several hours |
| IEMOCAP preprocessing (cut 7,380 clips from 151 dialogue-level source files) | ~15–20 min, 16-way parallel |
| IEMOCAP feature caching | under 3 minutes (train 84.8s, val 30.0s, test 32.0s — smaller corpus than MELD) |

No multi-GPU or distributed training was used anywhere in this project;
every run is single-GPU, single-process (`ProcessPoolExecutor` is used
only for CPU-side preprocessing, not training).

## Dataset acquisition

- **MELD**: public. `scripts/preprocess_meld.py --raw-dir <path>` expects
  the standard MELD.Raw release layout (see the script's own docstring
  and `docs/RECIPE.md`).
- **IEMOCAP**: gated by a SAIL (USC) usage license — **not public**.
  Contributors must obtain their own copy via
  https://sail.usc.edu/iemocap/ and agree to its release terms (cited in
  `docs/PAPER_SKELETON.md`'s compliance checklist: Busso et al. 2008
  citation, SAIL acknowledgment, clause-6 courtesy summary to SAIL before
  public reporting). `scripts/preprocess_iemocap.py --raw-dir <path>`
  expects the release's own `Session{1..5}/dialog/{wav,avi,transcriptions,
  EmoEvaluation}` layout unchanged — see `docs/iemocap_inventory.md` for
  the exact structure this repo verified against a real license-holder's
  copy (md5-checked release; hash recorded there, not reproduced here
  since it identifies a specific licensed distribution).

Neither dataset's raw or processed files are committed to this repo
(`data/` is gitignored); only the derived `metrics.json` results
(`paper_assets/results_archive/`) and the code that produced them are.

## What's NOT included in this repo, and why

| artifact | why not committed | how to get it |
|---|---|---|
| `outputs/*/best_model.pt` (checkpoints) | large binaries (tens of MB each x ~60 runs), no standard git-LFS setup in this repo | re-run the relevant `scripts/train_*.py` (Tier 3), or request from the authors |
| `data/meld/`, `data/iemocap/` (raw + processed + feature caches) | large, and IEMOCAP specifically is license-gated (can't be redistributed) | MELD: public download + `scripts/preprocess_meld.py`. IEMOCAP: obtain your own SAIL license, then `scripts/preprocess_iemocap.py` |
| `outputs/*/pip_freeze.txt`, `git_commit.txt` | per-run provenance snapshots, present in the real `outputs/` but not archived (not needed by any table/figure script; the repo's own commit history is the provenance record) | present in a real training run's output directory |
