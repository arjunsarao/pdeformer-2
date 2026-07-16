# Experiment log

This document summarizes every job represented in `slurm_logs/` as of 2026-07-10. Metrics are mean relative evaluation error (`eval_error_mean`); lower is better. A value near 1 means that the prediction is no better than a near-zero/mean-like prediction under this metric. “Best” is the value printed by the training script, not a value reconstructed after the fact.

## 2026-07-10 adapter audit

Job 10892 completed after downloading the local dataset. It trained on the
`train` split and evaluated on the `train` split, reaching train error 0.6412
and best test error 1.0207 at epoch 4. This did **not** constitute a same-example
or representative same-split test: the generic single-PDE wrapper took the
first 64 consecutive Well windows for training and reversed the 128 evaluation
indices from the end of the 960,000-window split. The small training subset was
therefore drawn from one trajectory/file, while evaluation came from the other
end of the split.

The adapter audit found and fixed three confounders before the next experiment:

1. Small Well subsets now use a deterministic split-wide permutation. For the
   64-sample Gray–Scott configuration this selects 10–11 examples from each of
   the six parameter files instead of 64 adjacent windows from one trajectory.
2. Multi-input forecasts now use lead time relative to the final input frame,
   matching The Well preprocessing contract and PDEformer's `t=0` initial
   condition convention.
3. The Gray–Scott equation DAG is now transformed consistently with z-score or
   RMS field normalization. Previously normalized `A` and `B` were inserted
   into the unnormalized physical equation, so its reaction and feed/kill terms
   described a different PDE.

The standalone evaluator now uses deterministic split-wide example selection
and reports persistence, normalized-mean, and physical-zero baselines alongside
PDEformer. Run this audit on the validation split before another fine-tuning
sweep.

## Executive summary

The standard, repository-native datasets show that the model and fine-tuning loop can learn:

| Dataset | Completed job | Epoch 0 test | Best test | Final test | Result |
|---|---:|---:|---:|---:|---|
| Burgers 2D viscous | 7599 | 0.2673 | 0.1576 | 0.1613 | Learns; mild overfitting after epoch 15 |
| Allen–Cahn 2D | 7633 | 0.9587 | 0.2567 | 0.2567 | Learns strongly and was still improving |
| Gray–Scott 2D (native loader) | 8609 | 0.2394 | 0.1089 | 0.1089 | Best result; learns and generalizes |
| Viscoelastic 2D | 8612 | 0.8512 | 0.3203 | 0.3457 | Learns; best checkpoint precedes epoch 20 |
| Gray–Scott (The Well, z-score) | 9286 | 0.9984 | 0.9984 | 1.0310 | Fits train (0.1275) but does not generalize |
| Gray–Scott (The Well, no normalization) | 9726 | 2.3165 | 2.0730 | 2.0730 | Wrong scale/representation; very poor |

The principal problem is therefore not generic PDEformer optimization. It is specific to the The Well integration and evaluation split. With z-score normalization, training error falls dramatically while validation stays at approximately 1 or becomes worse. Without normalization, both training and validation are very poor. The earlier runs were confounded by index selection: “64 samples” meant the first 64 adjacent windows of the split, not 64 samples per file, while test mode read from the opposite end. The local Gray–Scott train and validation directories contain the same six `(F, k)` regimes but different held-out data files/trajectories.

## Standard fine-tuning jobs

All successful jobs below use model-L (82.65M parameters), checkpoint `model-L.pt`, batch size 1, normalized RMSE, Adam, gradient clipping at 1, initial learning rate `5e-6`, 2,048 sampled points, and 20 epochs unless noted.

| Job | Dataset/config | Status | Epoch 0 test | Best test | Final test | Train at final | Notes |
|---:|---|---|---:|---:|---:|---:|---|
| 7598 | `burgers2d_viscous_model-L.yaml` | Failed after epoch 5 | 0.2673 | — | — | — | Evaluation/plotting crashed in NumPy quantile on an empty array |
| 7599 | `burgers2d_viscous_model-L.yaml` | Completed | 0.2673 | 0.1576 | 0.1613 | 0.0679 | Test best at epoch 15; subsequent train improvement did not generalize |
| 7630 | `allen_cahn2d_model-L.yaml` | Failed before evaluation | — | — | — | — | Missing `data/allen_cahn2d/allen_cahn2d.hdf5` |
| 7633 | `allen_cahn2d_model-L.yaml` | Completed | 0.9587 | 0.2567 | 0.2567 | 0.1625 | Strong consistent improvement through epoch 20 |
| 8609 | `grayscott2d_model-L.yaml` | Completed | 0.2394 | 0.1089 | 0.1089 | 0.0895 | Small train/test gap; strongest completed result |
| 8612 | `viscoelastic2d_model-L.yaml` | Completed | 0.8512 | 0.3203 | 0.3457 | unavailable in summary | Some late overfitting; select the best checkpoint |
| 8611 | Viscoelastic data job | No experiment output | — | — | — | — | Log contains only environment activation; no training result |

## The Well Gray–Scott jobs

Unless stated otherwise, these jobs use model-L, `model-L.pt`, batch size 1, 4,096 sampled points, one input step, one output step, stride 1, and remote data at `hf://datasets/polymathic-ai/`. `Fields=1` means field index `[0]`; `Fields=2` means `[0, 1]`. Later code explicitly requires both Gray–Scott fields, so the one-field runs are historical/structurally incomplete representations of the coupled system.

| Job | Samples train/test | Fields | Normalization; coordinates/time | LR; epochs | Status | Epoch 0 test | Best test | Final test | Key outcome |
|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 9257 | 4/4 | 1 | z-score; normalized | `5e-6`; 5 | Failed | — | — | — | `huggingface_hub` missing |
| 9258 | 4/4 | 1 | z-score; normalized | `5e-6`; 5 | Failed | — | — | — | Incomplete inherited config: `model.function_encoder` missing |
| 9259 | 4/4 | 1 | z-score; normalized | `5e-6`; 5 | Failed | — | — | — | Interpolation dimensionality error (128 grid points vs 1 value) |
| 9260 | 4/4 | 1 | z-score; normalized | `5e-6`; 5 | Failed | — | — | — | Coordinate/label rank mismatch (4-D vs 5-D) |
| 9261 | 4/4 | 1 | z-score; normalized | `5e-6`; 5 | Completed | 1.0271 | 1.0004 | 1.0004 | Tiny-data run reaches only the trivial-error regime |
| 9268 | 64/16 | 1 | z-score; normalized | `5e-6`; 20 | Completed | 1.0272 | 1.0272 | 1.0634 | Validation degrades with training |
| 9286 | 64/16 | 2 | z-score; normalized | `5e-6`; 20 | Completed | 0.9984 | 0.9984 | 1.0310 | Train reaches 0.1275 but validation remains trivial: severe split overfit/domain shift |
| 9718 | 64/16 | 1 | z-score; normalized | `1e-6`; 5 | Cancelled after epoch 0 | 1.0272 | — | — | Train at epoch 0 was 0.6136 |
| 9719 | 64/16 | 1 | z-score; normalized | `1e-6`; 5 | Cancelled during setup | — | — | — | No metric |
| 9720 | 64/16 | 2 | z-score; normalized | `1e-6`; 5 | Completed | 0.9984 | 0.9984 | 0.9992 | Train improves 0.7480 → 0.4053; validation does not move |
| 9722 | 64/16 | 1 | none; raw | `1e-6`; 5 | Failed | — | — | — | Two scalar values cannot map to a DAG with zero scalar slots |
| 9723 | 64/16 | 1 | none; raw | `1e-6`; 5 | Failed | — | — | Same DAG scalar shape mismatch |
| 9724 | 64/16 | 1 | none; raw | `1e-6`; 5 | Failed | — | — | Same DAG scalar shape mismatch |
| 9725 | 64/16 | 2 | none; raw | `1e-6`; 5 | Completed | 2.3165 | 2.2869 | 2.2869 | Almost no learning; train error remains 2.8712 |
| 9726 | 64/16 | 2 | none; raw | `5e-6`; 20 | Completed | 2.3165 | 2.0730 | 2.0730 | Longer/higher-LR training helps slightly but remains unusable; train 2.6996 |
| 10600 | 64/16 | 2 | z-score; normalized | `1e-5`; 20 | Cancelled during setup/data access | — | — | — | No metric |
| 10601 | 64/16 | 2 | z-score; normalized | `1e-5`; 20 | Completed | 1.0463 | 1.0463 | 1.0778 | Higher LR worsens validation; train reaches about 0.105 |
| 10638 | 64/16 | 2 | z-score; normalized | `1e-6`; 5 | Completed | 1.0463 | 1.0388 | 1.0558 | Evaluation fluctuates; no meaningful generalization |
| 10672 | 64/16 | 2 | z-score; normalized | `1e-6`; 5 | Cancelled during setup/data access | — | — | — | Seed fixed and RNG preservation enabled, but no metric |
| 10673 | 64/16 | 2 | z-score; normalized | `1e-6`; 5 | Completed | 1.0463 | 1.0463 | 1.0543 | Fixed seed/RNG does not solve the issue |
| 10867 | 64/128 | 2 | z-score; normalized | `1e-6`; 0 | Evaluation-only | 1.0446 | — | 1.0446 | Larger validation sample confirms error ≈1; not a 16-sample artifact |

The epoch-0 variation between otherwise similar jobs (approximately 0.998, 1.027, and 1.046) is consistent with changes in field selection, sampled files/examples, or code revisions. It does not alter the conclusion: every two-field z-score The Well run stays around relative error 1 on the validation split.

## Diagnosis

### 1. The Well train/validation failure is a generalization failure, not an inability to fit

Job 9286 is the clearest controlled result. Training error improves to 0.1275 while validation changes from 0.9984 to 1.0310. Jobs 9720 and 10601 show the same pattern at lower and higher learning rates. More epochs or a larger learning rate therefore amplify fitting without fixing validation.

The likely causes are, in descending priority:

1. **Biased subset selection (confirmed and fixed after job 10892).** `num_samples_per_file.train: 64` selected the first 64 adjacent temporal windows from one trajectory/file, rather than sampling all six files. Test mode selected windows from the opposite end.
2. **Normalized-field/equation mismatch (confirmed and fixed after job 10892).** Z-scored `A` and `B` were used in an untransformed physical Gray–Scott DAG. The reaction and feed/kill terms were therefore mathematically inconsistent with the model inputs and labels.
3. **Representation mismatch with pretraining.** The native Gray–Scott loader reaches 0.1089, while The Well representation starts near 1. Coordinate/time conventions and one-step target semantics still differ from pretraining and need the baseline audit below.
4. **One-step target may be dominated by normalization/identity structure.** With one input and one output step at stride 1, a persistence baseline should be strong. If PDEformer is near 1 while persistence is much better, the adapter or target alignment is wrong rather than the model being underpowered.
5. **Evaluation instability and remote I/O obscure comparisons.** Logs contain repeated HDF5 HTTP range requests/timeouts. Several jobs were cancelled during setup, and only one validation dataset instance is evaluated (`dataset_per_type: 1`). This makes experiments slow and estimates noisy, although job 10867 shows that increasing examples alone does not fix the result.

### 2. Turning normalization off is not viable

Jobs 9725 and 9726 show raw-data validation error above 2 and train error near 2.7. Z-score normalization is essential, but its implementation/statistics must be validated across splits. Coordinate/time normalization should be ablated separately from field normalization; the raw runs changed all three at once and are therefore not a clean coordinate/time ablation.

### 3. The native Gray–Scott result is the critical positive control

Job 8609 reaches 0.1089 on a closely related PDE. This rules out “PDEformer cannot model Gray–Scott” and points directly toward The Well data semantics, split composition, or adapter code.

## Recommended next experiments

Run these in order. Stop the expensive model sweeps until experiments 1–3 pass.

### 1. Baseline and alignment audit (no training)

Evaluate on exactly the same train and valid examples:

- zero prediction;
- per-channel training mean;
- persistence (`prediction = last input frame`);
- ground-truth target copied back through normalize → denormalize;
- target shifted by `-1`, `0`, and `+1` time indices;
- native-loader model/checkpoint, if inputs can be converted to the same physical convention.

Report error per field, per file/parameter pair, and per lead time. The normalize/denormalize identity must be near machine precision. A persistence baseline materially below 1 while PDEformer is near 1 is a high-confidence adapter/model-interface problem.

### 2. Overfit one fixed example and one fixed file

Use both fields and z-score normalization. Disable random example/point resampling, remote streaming, and augmentation. Try:

- one `(input, target)` pair until error is below 0.01;
- 16 pairs from one file, with an 8/8 random split within that same file;
- the same 16 pairs with shuffled targets as a negative control.

If a single pair cannot be fit, inspect tensor/field ordering, normalization, target-time indexing, DAG scalar placement, and whether gradients reach the function encoder/INR. If a same-file split works but the official valid split fails, the issue is distribution shift rather than optimization.

### 3. Split matrix to isolate distribution shift

Construct four evaluations with fixed examples:

| Train source | Test source | Purpose |
|---|---|---|
| Train files | held-out times from same files | Measures temporal interpolation |
| Train files | held-out trajectories from same parameters | Measures initial-condition generalization |
| Train parameters | held-out parameter files | Measures parameter extrapolation |
| Official train | official valid | Reproduces current result |

Log results per `(F, k)` file and pattern family (bubbles, worms, etc.), not only the global mean.

### 4. Clean preprocessing ablation

Keep field z-score normalization on and vary one factor at a time:

1. normalized coordinates + normalized time (current baseline);
2. raw coordinates + normalized time;
3. normalized coordinates + raw time;
4. global train-split statistics versus per-file statistics;
5. normalize the predicted delta `u(t+1)-u(t)` instead of the absolute next state.

Use a fixed seed, fixed cached files, fixed train/eval indices, and at least three seeds after identifying a promising setting.

### 5. Data coverage before capacity or LR sweeps

For the best verified preprocessing, sweep training coverage: 64, 256, and all available time pairs per file, while sampling parameter files uniformly. Compare model-S or frozen-backbone/decoder-only fine-tuning against model-L. The present 82.65M model with few examples is predisposed to memorization.

### 6. Optimization only after the pipeline passes

Use `1e-6`, `3e-6`, and `1e-5` with early stopping, but evaluate every epoch on a cached deterministic validation set. Add separate parameter groups: a smaller LR for the pretrained backbone and a 5–10× larger LR for newly initialized The Well/field-specific components. Track gradient norms and parameter-update norms by module to verify that the intended layers learn.

### 7. Multi-step and residual objectives

Once one-step performance beats persistence, compare:

- direct absolute next-state prediction;
- residual prediction `u(t+1) = u(t) + Δu`;
- strides 1, 2, 4, and 8;
- rollout horizons 1, 4, and 8 with horizon-weighted loss.

Residual prediction is the most promising first variant for small one-step changes.

## Experiment hygiene

- Cache The Well files locally before training; remote range requests and timeouts make runs expensive and non-reproducible.
- Give every job a unique `record_dir` and W&B name. Current The Well configs reuse the same paths/names, creating checkpoint/metadata collision risk.
- Save the resolved config, git commit, exact file list, example indices, normalization statistics, and checkpoint hash with every run.
- Keep `field_indices: [0, 1]` for Gray–Scott; the equation is coupled and current adapter code requires both fields.
- Evaluate more than `dataset_per_type: 1` and report confidence intervals across files/parameters and seeds.
- Always compare against persistence and a small convolutional/FNO baseline on the identical split. This distinguishes architecture limitations from dataset bugs.
