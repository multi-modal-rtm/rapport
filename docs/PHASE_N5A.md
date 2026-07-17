# PHASE N5-A — the subsumption curve (MELD)

Chart of context-window size k vs. what the relational/shift/temporal stack adds on top of the residual-redesigned fusion baseline (`docs/PHASE_N4R.md` spec v1.1). For k in {0, 2, 4, 8}: (1) retrained the Phase T text encoder end-to-end AT that k, caching its embeddings + own logits (`cache_version text_ctx_k{k}`); (2) trained `base_fusion_R` and `full_R` on that cache; (3) at the endpoints k=0 and k=8, `full_R` ("the graph config") was powered up to **n=7 seeds** at BOTH endpoints (42, 1337, 2024, 7, 123, 555, 9090) -- all downstream reruns on the same shared per-k cache, per the original 3-seed convention. The text anchor reaches n=7 at k=8 (Phase T's existing 3 seeds + 4 new retrains) but only **n=5 at k=0** (42, 7, 123, 555, 9090): the k=0 text anchor was seed-42-only in the original design, and this consolidation pass added exactly the 4 new seeds it was asked to add, not 6. `base_fusion_R` stays at n=3 (seeds 42/1337/2024, not powered up in this consolidation pass).

![subsumption curve](subsumption_curve.png)

## Reconciliation: N4-R Step 2's +0.0147 vs. this phase's +0.0025 at k=0

These are **not two measurements of the same quantity** -- every one of
four things differs between them:

| | N4-R Step 2 (`docs/PHASE_N4R.md`) | N5-A (this doc) |
|---|---|---|
| **Architecture** | `base_fusion` -- the ORIGINAL, pre-residual Phase N4 architecture (`residual` flag didn't exist yet; Step 2 ran before Step 3's redesign) | `base_fusion_R` / `full_R` -- the residual redesign (spec v1.1: zero-init `W_out`, zero-init fusion A/V blocks) |
| **k=0 text encoder** | The SAME k=8-TRAINED Phase T checkpoint (`context_text_plain_ce_seed42`), fed context-free input only AT INFERENCE TIME when building the cache -- never retrained | A GENUINELY freshly fine-tuned encoder, LoRA-trained end-to-end with k=0 context windows during its OWN training (`context_text_k0_seed42`) |
| **Gain definition** | `base_fusion` (which already contains the v1-style per-speaker GAT+GRU -- there is no "graph-free" baseline in the original architecture) **minus** the text-only anchor | `full_R` (relational memory + shift + temporal, on top of `base_fusion_R`'s own GAT+GRU) **minus** `base_fusion_R` (fusion + GAT+GRU alone, no relational memory/shift/temporal) |
| **Seeds** | 1 (seed 42) | 3, now 7 at the endpoints (seeds 42/1337/2024/7/123/555/9090) |

**They measure different things by design, not by accident.** Step 2 was
an explicitly-labeled "one cheap run" triage diagnostic (Phase N4-R's own
words) to decide *whether the residual redesign was worth building at
all* -- it asked "does the whole fusion+graph stack beat text alone more
when text has no context of its own," using whatever encoder was already
on hand (inference-time context removal, not a real retrain, since
retraining a whole new encoder wasn't yet justified at that stage). N5-A
asks the narrower, better-controlled question that Phase N4-R's own
conclusion motivated: "does the INCREMENTAL relational/shift/temporal
machinery earn its keep over a matched fusion-only baseline, when the
encoder itself was genuinely trained to have no cross-utterance context" --
using the current (post-redesign) architecture throughout.

**For completeness, the literal analog of Step 2's definition recomputed
on N5-A's assets** (`base_fusion_R` minus the genuinely-k0-trained text
anchor, at k=0, 3-seed `base_fusion_R` mean vs. seed-42 text anchor):
0.6279 - 0.6395 = **-0.0116** -- the opposite sign from Step 2's +0.0147.
This is a real, informative difference, not noise: it shows that once the
encoder is actually retrained at k=0 (rather than just fed k=0 input
through k=8-trained weights), the "text-only" anchor itself is
considerably stronger (0.6395 vs. Step 2's inference-only anchor of
0.6234) -- strong enough that `base_fusion_R`'s fusion+graph stack no
longer beats it at all. Step 2's inference-only anchor understated how
good a properly-trained context-free encoder actually is.

**Which number the paper reports: N5-A's (the n=7 `full_R` vs. text-anchor
paired comparison below), not Step 2's.** N5-A uses the current
architecture (the one Phase N4-R actually shipped), a methodologically
correct encoder construction (genuinely trained at k=0, not
inference-time-perturbed), and multiple seeds rather than 1. Step 2's
number stays on the record as the preliminary diagnostic that motivated
building N5-A properly, not as a competing or superseded-by-noise result
-- it was never intended to be the paper's statistic.

## Data (mean ± std; single point where n_seeds=1)

| k | text anchor | base_fusion_R | full_R | n_seeds (anchor/full_R, base_fusion_R) |
|---|---|---|---|---|
| 0 | 0.6352±0.0033 | 0.6279±0.0012 | 0.6274±0.0114 | 5/7, 3 |
| 2 | 0.6192±0.0000 | 0.6375±0.0000 | 0.6374±0.0000 | 1/1, 1 |
| 4 | 0.6329±0.0000 | 0.6400±0.0000 | 0.6146±0.0000 | 1/1, 1 |
| 8 | 0.6431±0.0053 | 0.6392±0.0011 | 0.6344±0.0043 | 7/7, 3 |

## Graph's marginal gain per k (full_R − base_fusion_R, unpaired means)

| k | gain | n_seeds (full_R) |
|---|---|---|
| 0 | -0.0005 | 7 |
| 2 | -0.0000 | 1 |
| 4 | -0.0254 | 1 |
| 8 | -0.0048 | 7 |

## PRE-REGISTERED test: paired (per-seed) full_R − text anchor, at the endpoints

Paired over the intersection of seeds present in both series. At k=8 both series have all 7 seeds, so n_pairs=7. At k=0 the text anchor only has 5 distinct encoder retrains (see above), so seeds 1337 and 2024 -- which have a `full_R` downstream run but no matching k=0 text-anchor retrain under that seed label -- are excluded rather than paired against a different seed's anchor value, giving n_pairs=5 at k=0.

| k | n_pairs | per-seed diffs (seed: diff) | mean | std | |diff|>std? |
|---|---|---|---|---|---|
| 0 | 5 | 7:-0.0008, 42:-0.0040, 123:-0.0020, 555:-0.0322, 9090:-0.0008 | -0.0080 | 0.0136 | False |
| 8 | 7 | 7:-0.0118, 42:+0.0021, 123:-0.0174, 555:-0.0049, 1337:-0.0168, 2024:-0.0031, 9090:-0.0094 | -0.0088 | 0.0072 | True |

## PRE-REGISTERED DECISION (fixed before this result was computed)

Decided on the k=0 paired result at n_pairs=5 (5, not 7 -- see the paired-test note above on why 1337/2024 are excluded at k=0).

**k=0 paired gain (-0.0080) is within noise (std 0.0136): THE STRONGER NULL.** Even in the one condition most favorable to the graph -- a text encoder with little to no context of its own -- `full_R` does not measurably beat the text-only anchor. The paper's claim should be framed accordingly: the relational/shift/temporal machinery does not demonstrate a measurable benefit over a strong text baseline at ANY tested context window size, not just at the locked k=8 recipe.

## Legacy: monotonicity of the unpaired full_R − base_fusion_R gain

- Strictly monotonic decreasing across all 4 points: **False**
- Endpoints only (k=0 vs k=8, both robust means): gain shrinks from -0.0005 to -0.0048 -> **True**

## Reproducibility

`scripts/report_subsumption_curve.py` regenerates this ENTIRE doc (including the reconciliation section above, which is static authored text kept in the script) and the figure from `outputs/*/metrics.json` directly -- rerun any time after `scripts/run_subsumption_matrix.py` / `scripts/run_n7_powerup.py`. `outputs/subsumption_curve_data.json` has the full underlying per-seed data.