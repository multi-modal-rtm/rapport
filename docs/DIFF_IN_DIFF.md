# Difference-in-Differences: Frozen vs. Fine-Tuned Regime

Reviewer-response experiment (Major Concern #1): a formal interaction test for the boundary claim, computed instead of only comparing the two regimes' point estimates across separate tables. **No stored value was changed or recomputed by hand; this script reads only `outputs/subsumption_curve_data.json` (already used by `tab:frozen-bridging` and `tab:k-sweep-endpoints`) and derives everything below from it.**

## Method

Independent-groups percentile bootstrap. The frozen-regime paired per-seed gains (`bridging.paired_diffs`, n=3) and the fine-tuned-regime paired per-seed gains (`paired_n7_test.{k}.diffs`, n=7 at k=8 / n=5 at k=0) are two **independent** samples -- overlapping seed *values* across regimes (e.g. both include a seed-42 run) do not make them paired runs, since a frozen seed-42 run and a fine-tuned seed-42 run share nothing but an RNG initialization. Each of 100,000 bootstrap iterations resamples each group's per-seed diffs with replacement at its own original size and computes (resampled frozen mean) minus (resampled fine-tuned mean); the 2.5th and 97.5th percentiles of the resulting distribution form the reported 95% interval. Fixed RNG seed 20260818 for reproducibility (not a model-training seed).

## Result

The primary comparison uses the paper's headline fine-tuned comparator, the locked recipe at k=8 (n=7 seeds): the regime difference-in-differences is **+0.0444** (frozen gain minus fine-tuned gain), 95% bootstrap CI **[+0.0347, +0.0560]**, which excludes zero (100.00% of bootstrap resamples have frozen gain exceeding fine-tuned gain). This interval, not any single within-regime delta, is the paper's formal significance claim: it does not require the fine-tuned k=8 gain itself to be individually significant, only that the frozen-regime gain is reliably larger than the fine-tuned-regime gain.

As a robustness check against the context-free fine-tuned endpoint (k=0, n=5) instead of the locked recipe: DiD = **+0.0436**, 95% CI **[+0.0307, +0.0605]**, excludes zero. Both comparators agree in direction and in excluding zero.

## Table for Section IV insertion (prose/label integration deferred)

```latex
\begin{table}[t]
\centering
\caption{Regime difference-in-differences: frozen-regime paired gain minus fine-tuned-regime paired gain, independent-groups percentile bootstrap ($n{=}100,000$ resamples), 95\% interval.}
\label{tab:diff-in-diff}
\begin{tabular}{lccc}
\toprule
fine-tuned comparator & $n$ (frozen/f.t.) & DiD & 95\% CI \\
\midrule
$k=8$ (locked recipe) & 3/7 & +0.0444 & [+0.0347, +0.0560] \\
$k=0$ (context-free) & 3/5 & +0.0436 & [+0.0307, +0.0605] \\
\bottomrule
\end{tabular}
\end{table}
```

## One-paragraph result (drafted for later Section IV insertion, not yet inserted)

> The boundary claim admits a formal interaction test: the regime difference-in-differences (frozen-regime paired gain minus fine-tuned-regime paired gain, independent-groups percentile bootstrap, 100,000 resamples) is $\mathbf{+0.0444}$, 95\% CI $[+0.0347, +0.0560]$, excluding zero. This is the paper's real significance claim, and it does not require any single fine-tuned-regime delta to itself be significant: it requires only that the frozen-regime gain is reliably larger than the fine-tuned-regime gain, which the interval confirms.

## Raw inputs (for audit)

- Frozen paired diffs (n=3): `[0.04770054383288086, 0.031264971254471474, 0.027974502334780327]`

- Fine-tuned k=8 paired diffs (n=7): `[-0.011754123695041363, 0.0020822964685924816, -0.017374622385125882, -0.004902287235825642, -0.01682045074453964, -0.00311363085709937, -0.009418058831503151]`

- Fine-tuned k=0 paired diffs (n=5): `[-0.0007683198328711782, -0.0040295696320173935, -0.0020021013074479344, -0.032199583369822316, -0.0008246104059453918]`


**Caveat, stated plainly**: the frozen-regime sample is only n=3 seeds, so its bootstrap resampling draws from just 3 distinct values (10 possible multisets of size 3 from 3 items) -- the interval's width is driven largely by this small frozen-side sample, not by bootstrap resolution limits on the fine-tuned side. Treat the interval as a lower-resolution but still valid 95% interval, not as evidence of high precision on the frozen-regime estimate.
