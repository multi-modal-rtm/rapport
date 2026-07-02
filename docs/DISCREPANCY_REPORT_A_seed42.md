# v1 SocialArcNet Baseline Reproduction — Discrepancy Report

**Run:** config A, seed 42, `outputs/A_baseline_seed42/`
**Date:** 2026-07-02
**Verdict: GATE FAILED.** Test weighted F1 = **0.3765**, target range [0.60, 0.64]. Per instructions, no hyperparameter tuning was attempted to chase the number — reporting the discrepancy instead and stopping for sign-off.

## Result summary

| Metric | Value | Paper reference |
|---|---|---|
| Test weighted F1 | **0.3765** | 0.62 |
| Test accuracy | 0.4295 | 0.61 |
| Test macro F1 | **0.1840** | 0.57 |
| Best val weighted F1 | 0.4124 (epoch 29) | — |
| Epochs run | 40 (early-stopped, patience=10) | — |
| Avg epoch wall-clock | 8.37 s | — |
| Total training wall-clock | ~335 s (~5.6 min) | — |

All three metrics are far below the paper's numbers — this isn't a borderline miss, it's a large, consistent gap across weighted F1, accuracy, and (especially) macro F1. The macro F1 gap is the most severe (0.184 vs 0.57), pointing at systematic minority-class failure rather than generic underfitting.

## Learning curve

Loss decreases smoothly and monotonically for all 40 epochs (0.921 → 0.682) — training is stable, no divergence, no NaNs, no crashes. Val weighted F1 climbs from 0.25 (epoch 0, majority-class-only baseline) to a peak of 0.4124 at epoch 29, then plateaus/oscillates in the 0.35–0.41 band without further improvement, triggering early stopping at epoch 39.

```
epoch  loss    val_wF1  val_mF1   neutral  joy    sadness anger  surprise fear  disgust
0      0.9210  0.2518   0.0850    0.595    0.0    0.0     0.0    0.0      0.0   0.0
6      0.8149  0.2555   0.0887    0.595    0.0    0.0     0.025  0.0      0.0   0.0
13     0.7799  0.3038   0.1324    0.616    0.012  0.0     0.298  0.0      0.0   0.0
16     0.7644  0.3476   0.1737    0.632    0.185  0.053   0.250  0.096    0.0   0.0
23     0.7360  0.3628   0.1880    0.639    0.223  0.070   0.223  0.161    0.0   0.0
29     0.7154  0.4124   0.2350    0.652    0.287  0.068   0.375  0.262    0.0   0.0   <- best val, checkpointed
33     0.7034  0.4082   0.2303    0.651    0.268  0.052   0.375  0.266    0.0   0.0
37     0.6854  0.4103   0.2289    0.661    0.285  0.035   0.304  0.318    0.0   0.0
39     0.6819  0.3946   0.2147    0.657    0.203  0.035   0.318  0.291    0.0   0.0   <- early stop
```

**`fear` and `disgust` val F1 are exactly 0.0 for every single one of the 40 epochs.** The model never once predicts either class on the validation set, from initialization through convergence. Full per-epoch history: `outputs/A_baseline_seed42/tensorboard/` and `data/meld/train_A_seed42.log`.

## Test-set confusion matrix

![confusion matrix](../outputs/A_baseline_seed42/confusion_matrix.png)

Rows = true label, columns = predicted, order [neutral, joy, sadness, anger, surprise, fear, disgust].

- **`fear` and `disgust` columns are entirely zero** — not one test utterance was ever predicted as either class, matching the validation-time collapse.
- `neutral` is the dominant catch-all (938/1256 correctly, but also absorbs most of the mass from every other true class: 248 of joy's 402, 199 of anger's 345, 181 of surprise's 281, 122 of sadness's 208, 32 of fear's 50, 35 of disgust's 68).
- `sadness` acts as a **second catch-all**, disproportionately absorbing misclassifications from classes that aren't semantically close to sadness (212 from true-neutral, 64 from true-joy, 55 from true-anger, 50 from true-surprise) while its own recall is mediocre (72/208 = 35%). This pattern — a mid-frequency class soaking up cross-class confusion broadly rather than just similar emotions — is consistent with the model latching onto a shallow, low-information decision boundary rather than genuinely separating the V/A/T feature space by emotion.

I do not have access to the actual paper's confusion matrix to compare directly (ICFNDS '25 isn't a paper I could retrieve/verify against in this environment) — this analysis is based on internal consistency (label ordering verified correct end-to-end; `neutral` support of 1256 matches MELD's published test distribution) plus general priors about what a working ERC baseline's error pattern should look like (errors concentrated between semantically adjacent emotions, not blanket collapse onto 2 of 7 classes).

## Root-cause hypotheses, ranked by suspicion

I did not act on any of these — flagging for your sign-off before touching hyperparameters or architecture, per the gate instructions.

1. **RoBERTa's `pooler_output` is randomly initialized (high suspicion).** `roberta-base`'s HF checkpoint ships *without* pretrained pooler weights (BERT has one from NSP pretraining; RoBERTa's pretraining objective dropped it, so HF initializes `pooler.dense.{weight,bias}` fresh on load — confirmed via the "MISSING" load report when the backbone was built). Since the backbone is frozen, the text modality's pooled 768-d feature is a **fixed random projection** of the [CLS] hidden state, not a learned representation — likely destroying most of the text signal, which is typically the strongest single modality for MELD ERC. I followed the task spec literally ("the pooler ([CLS]) output for text"), but this is worth revisiting — e.g. mean-pooling `last_hidden_state` or using the raw (pretrained) [CLS] token directly instead of the HF pooler.
2. **MViTv2 frame-count mismatch, worked around via upsampling.** The Kinetics-pretrained positional encoding is fixed for 16-frame clips; our preprocessing (per the prior phase's spec) stored 8 uniformly-sampled frames. I repeat-interleaved 8→16 frames inside the backbone wrapper to satisfy the model's positional encoding shape, rather than re-extracting 16 native frames. This is a plausible, secondary source of degraded video features (duplicated frames carry no new temporal information vs. genuinely re-sampling 16).
3. **Minority-class collapse despite focal loss (gamma=3).** `fear` (268 train / 50 test) and `disgust` (271 train / 68 test) are MELD's two smallest classes by a wide margin (next smallest, `sadness`, has 683/208) and get exactly zero recall throughout training. Focal loss down-weights *easy* examples but doesn't guarantee minority-class attention on its own, especially over a ~2% base rate. Possible next step (pending sign-off): class-weighted loss on top of focal, or verifying the focal loss reduction/implementation isn't itself under-weighting rare-class gradients more than intended.
4. **Architecture-spec ambiguity.** Several Social GNN design choices were necessarily interpreted rather than verified against source (no paper/reference code was available to check against): fully-connected speaker graph (vs. a windowed/partial graph), GAT operating on hidden states rather than fused inputs, single GAT layer, dropout applied in both GAT attention and the classifier head. Any of these could diverge from the actual published architecture and individually shift results.
5. **Untuned batch size.** The task spec didn't fix a batch size; I used 16 dialogues/batch. Given training loss behaves smoothly, this is low on my suspicion list, but noting it for completeness.

## Artifacts

- `outputs/A_baseline_seed42/classification_report.json` — full precision/recall/F1 per class
- `outputs/A_baseline_seed42/confusion_matrix.png`
- `outputs/A_baseline_seed42/resolved_config.yaml`, `git_commit.txt`, `pip_freeze.txt`
- `outputs/A_baseline_seed42/tensorboard/` — full scalar history (loss, val weighted/macro F1, per-class F1, LR)
- `outputs/A_baseline_seed42/best_model.pt` — checkpoint at best val epoch (29)
- `data/meld/train_A_seed42.log` — raw stdout/epoch log

## Wall-clock

Avg **8.37 s/epoch** on the RTX 5090 (bf16 autocast, cached 768-d features — no backbone forward passes during training). 40 epochs → ~335 s total. Feature-cache build (one-time, all 13,706 utterances × 3 frozen backbones) took ~215 s for train+dev, plus additional time for test after working around a CUDA allocator fragmentation issue during that run (see below) — not part of the per-epoch training budget.

## Note: infrastructure issue encountered and fixed during this phase

The feature-cache build hit two real bugs, both now fixed in `scripts/build_feature_cache.py` and `MELDCachedDataset`:
1. **Cache key collision across splits** — `dialogue_id` restarts from 0 in each MELD split (train/dev/test), so the initial flat `cache/{modality}/dia{X}_utt{Y}.pt` layout silently overwrote files across splits (13,706 expected, only 11,132 landed on disk). Fixed by namespacing cache paths under `cache/{modality}/{split}/...`.
2. **CUDA allocator fragmentation** from highly variable per-batch audio sequence lengths eventually exhausted the GPU mid-run (process alone reached 31.2/31.36 GiB). Mitigated via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, sorting each split by audio duration before batching, an OOM-safe batch-halving retry, and making the cache builder resumable (skips already-cached utterances) so a re-launch always makes forward progress. This is worth keeping in mind for Phase-3-scale (or larger) cache builds later.

These are infrastructure fixes, not modeling changes, and are unrelated to the GATE failure above.
