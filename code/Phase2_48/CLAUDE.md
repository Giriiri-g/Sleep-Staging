# CLAUDE.md
## Explainable Deep Learning for Sleep Disorder Detection and Personalised Lifestyle Recommendations

> **For AI agents**: This file is the single source of truth for all context about this project. Read it fully before making any changes to code, notebooks, or documentation. Every section is load-bearing.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Research Objectives and Questions](#2-research-objectives-and-questions)
3. [Dataset](#3-dataset)
4. [Experiment History — Complete Record](#4-experiment-history--complete-record)
5. [Model Architecture — Temporal-Spectral Fusion](#5-model-architecture--temporal-spectral-fusion)
6. [Explainability Pipeline](#6-explainability-pipeline)
7. [File Map — Inputs and Outputs](#7-file-map--inputs-and-outputs)
8. [SomnAI — Platform Architecture](#8-somn-ai--platform-architecture)
9. [Angular Application — Component and Service Map](#9-angular-application--component-and-service-map)
10. [Key Technical Decisions and Constraints](#10-key-technical-decisions-and-constraints)
11. [Known Issues and Pitfalls](#11-known-issues-and-pitfalls)
12. [Desired Goals and Open Research Gaps](#12-desired-goals-and-open-research-gaps)
13. [Conventions and Non-Negotiables](#13-conventions-and-non-negotiables)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project name** | Project 48 |
| **Full title** | Explainable Deep Learning Approach for Sleep Disorder Detection and Personalised Lifestyle Recommendations |
| **Academic guide** | Prof. Dr. Simi Surendran |
| **Platform name** | SomnAI — Sleep Intelligence Platform |
| **Dataset** | MESA Sleep Study (Sleep-EDF-x v1.0.0, pre-processed) |
| **Primary language** | Python (research/ML), TypeScript/Angular 21 (platform) |
| **Primary compute** | Kaggle GPU (P100 / T4) |
| **Repository structure** | Kaggle notebooks + Angular workspace (`sleep-explainer/`) |

**In one sentence**: A multi-disorder sleep staging system that classifies PSG recordings into five sleep stages (W, N1, N2, N3, REM), detects three disorders (Apnea, Insomnia, RLS), and explains every prediction to both clinicians (via EEG signal event localisation and confidence scoring) and patients (via plain-English risk scores, feature importance, counterfactuals, and an AI advisor chat).

---

## 2. Research Objectives and Questions

### Primary objectives

1. Achieve clinically meaningful F1 scores for all three disorder targets simultaneously — not just the easiest (Apnea).
2. Build a feature selection pipeline that identifies complementary feature groups, not just standalone strong predictors (the core failure of L1/Lasso approaches).
3. Produce per-epoch, per-feature explainability that is interpretable by a non-technical patient without clinical training.
4. Quantify prediction uncertainty (MC-Dropout entropy) per epoch to flag cases that require human review.

### Research questions

- **RQ1**: Can a temporal-spectral fusion model outperform individual-modality models for multi-disorder sleep staging?
- **RQ2**: Does ElasticNet bootstrap stability selection (combining L1 sparsity + L2 group retention) outperform pure L1 for discovering complementary feature sets in questionnaire data?
- **RQ3**: Is there a label-feature alignment ceiling that limits PSG signal features for self-reported disorder labels?
- **RQ4**: Does merging PSG signal features with questionnaire features yield complementary gains, and what feature selection approach is required to realise those gains?
- **RQ5**: Can GradCAM + attention rollout produce clinically interpretable epoch-level EEG event localisation?

### Confirmed findings (as of Experiment 6b)

- **RQ2 confirmed**: ElasticNet raised Insomnia F1 from 0.286 (L1-only, Exp 3) to 0.388 (Exp 4) — +35.9% relative improvement.
- **RQ3 confirmed**: Self-reported Apnea label trained on questionnaire features F1=0.784. Same label trained on pure PSG signals F1=0.183. The label encodes reporting behaviour, not raw physiology.
- **RQ4 partially confirmed**: The merge is sound but requires a two-stage MI pre-filter before stability selection. Without it, Experiment 6 catastrophically failed (95% feature retention, dimensionality collapse). Experiment 6b corrected this — results pending.
- **RQ1 pending full Temporal-Spectral Fusion evaluation**: The fusion notebook trains successfully on Kaggle GPU. Final cross-disorder comparison not yet complete.

---

## 3. Dataset

### Source
- **MESA Sleep Study** (Multi-Ethnic Study of Atherosclerosis)
- Accessed via: `girishgiriirig/sleep-edfx-v1-0-0-eeg-preprocessed` on Kaggle
- Pre-processing pipeline produces two distinct CSVs:

### Dataset A — Questionnaire CSV
| Property | Value |
|---|---|
| **File** | `mesa-sleep-dataset-0.8.0.csv` |
| **Path (Kaggle)** | `/kaggle/input/mesa-sleep-dataset/mesa-sleep-dataset-0.8.0.csv` |
| **Shape** | 2,237 rows × 628 columns (raw), after dropping >40% missing → ~570 usable |
| **ID column** | `mesaid` (integer, subject identifier) |
| **Target columns** | `insmnia5` (Insomnia), `rstlesslgs5` (RLS), `slpapnea5` (Apnea) |
| **Target type** | Binary integer (0/1), self-reported questionnaire answers |
| **Positive rates** | Insomnia: 6.3% · RLS: 4.5% · Apnea: 7.6–8.9% |
| **Feature types** | Demographics, medical history, questionnaire answers, sleep diary, actigraphy summary, calculated risk scores, PSG summary statistics |
| **Critical note** | Targets are **self-reported diagnoses** — they encode whether a subject was *told* they have the disorder, not a clinically measured physiological value. This is the label-feature alignment ceiling. |

### Dataset B — PSG Signal-Derived CSV
| Property | Value |
|---|---|
| **File** | `output.csv` |
| **Path (local/uploads)** | `/mnt/user-data/uploads/output.csv` |
| **Shape** | 2,056 rows × 136 columns |
| **ID column** | `nsrrid` (integer, = `mesaid` for the same subjects) |
| **Target columns** | `Insomnia`, `RLS`, `apnea` (binary integer, computed from same MESA labels) |
| **Positive rates** | Insomnia: 6.3% · RLS: 4.5% · Apnea: 7.6% |
| **Feature groups** | Sleep architecture (24 cols) · Respiratory/Apnea (20 cols) · EEG spectral power per band per stage (60 cols) · HRV (7 cols) · MeanHR per stage (5 cols, mostly sparse) · Leg Movement/PLM (6 cols) · REM/Bruxism EMG (9 cols) · SpO₂/desaturation (8 cols) |
| **Zero missing values** | True — pre-processing pipeline guarantees complete rows |
| **Known data quality issues** | (1) HRV zeros = failed computation, not true zero → must replace with NaN then impute. (2) MeanHR per stage: 91–95% zero → drop entirely. (3) `EEG1_N2_Sigma_mean` == `EEG1_N2_Sigma_Power_mean` exact duplicate → drop one. (4) 28 highly correlated pairs \|r\| > 0.95 → prune one of each. |

### Subject overlap
- `nsrrid` (PSG CSV) == `mesaid` (questionnaire CSV) — same subjects, same identifier values.
- Inner join on these columns retains all 2,056 PSG subjects (all present in the larger questionnaire set).
- Merged shape before preprocessing: 2,056 rows × ~757 columns.

### Sleep stage labels (temporal-spectral fusion model)
- **Format**: Per-epoch labels, 30-second windows, stored as `.pt` tensors alongside raw EEG.
- **Stage mapping**: `W=0, N1=1, N2=2, N3=3, REM=4`
- **Typical recording**: ~180 epochs = 90 minutes. Window size used: 256 epochs per chunk.

---

## 4. Experiment History — Complete Record

### Primary metric
**F1 score and PR-AUC** are the correct metrics for this task. ROC-AUC is misleading at 4–8% positive rates — a classifier predicting all-negative scores AUROC=0.5 but F1=0. Use F1 as the headline number.

---

### Experiment 1 — RNN Baseline (Phase 1 existing pipeline)
| Property | Value |
|---|---|
| **Notebook** | (Phase 1, pre-Project 48) |
| **Features** | 9 macro sleep architecture features extracted manually from PSG + physiological CSV |
| **Model** | RNN-based embedding extraction |
| **Metric** | ROC-AUC |
| **Results** | Apnea=0.614 · RLS=0.540 · Insomnia=0.568 |
| **Finding** | Near-random for RLS and Insomnia. Apnea has moderate discriminative ability. |
| **Status** | Baseline only — not included in final presentations |

---

### Experiment 2 — Manual Sleep Architecture Features
| Property | Value |
|---|---|
| **Features** | 9 clinically-chosen features: WASO, %N1, %N2, %N3/4, %REM, TIB, TST, Sleep Latency, REM Latency |
| **Models** | SVM, XGBoost, Random Forest, Logistic Regression, MLP |
| **Tuning** | Threshold fine-tuned to maximise F1 from OOF probabilities |
| **Metric** | ROC-AUC |
| **Results** | Apnea=0.757 (+14%) · RLS=0.622 (+6%) · Insomnia=0.629 (+8%) |
| **Finding** | Manual clinical features outperform RNN baseline. Apnea shows meaningful improvement. But features were chosen by clinical intuition without analytical justification. |

---

### Experiment 3 — Automated Feature Selection (L1/MI/XGBoost)
| Property | Value |
|---|---|
| **Notebook** | `project48_feature_study_v3.ipynb` |
| **Input** | Full 628-column questionnaire CSV |
| **Pipeline** | Missing threshold drop → variance filter → correlation pruning (\|r\|>0.95) → MI ranking → XGBoost importance → L1 Logistic stability selection |
| **Features retained** | 9 stable features (AHI, WASO, BMI, Sleep Bouts/day, Weight, Max HR, Central Apnea Index, Hypopnea%, CPAP used) |
| **Models** | LogisticReg, RandomForest, XGBoost, LinearSVM, MLP_sklearn, MLP_PyTorch |
| **Metric** | F1 |
| **Results** | Apnea=0.782 · Insomnia=0.286 · RLS=0.166 |
| **Critical finding** | L1/Lasso selects only **standalone strong predictors** — it systematically discards features that are individually weak but jointly powerful. Insomnia and RLS require complementary feature sets that Lasso misses. |

---

### Experiment 4 — ElasticNet Bootstrap Stability Selection ⭐ BEST QUESTIONNAIRE RESULT
| Property | Value |
|---|---|
| **Notebook** | `project48_feature_study_v3_optimised.ipynb` |
| **Input** | Full 628-column questionnaire CSV |
| **Key change** | Replaced L1-only with ElasticNet (L1+L2). L2 encourages correlated features to stay together, capturing complementary feature groups that L1 discards. |
| **Pipeline** | Same preprocessing → ElasticNet bootstrap stability (30 bootstraps, C=0.1, l1_ratio=0.5, threshold=0.3) |
| **Features retained** | 326–383 per target (52–61% retention — noted as too permissive in hindsight) |
| **Models** | LogisticReg, RandomForest, XGBoost, LinearSVM, MLP_sklearn, MLP_PyTorch (256→128→64→1, BCEWithLogitsLoss+pos_weight) |
| **Runtime** | ~38.2 min on Kaggle CPU |
| **Results** | |
| Apnea | F1=0.7843 (XGBoost) · AUROC=0.8835 · PR-AUC=0.7646 |
| Insomnia | F1=0.3881 (MLP_PyTorch) · AUROC=0.7597 · PR-AUC=0.1944 |
| RLS | F1=0.1553 (MLP_PyTorch) · AUROC=0.6308 · PR-AUC=0.0934 |
| **Notable anomalies** | LinearSVM Apnea AUROC=0.9182 but F1=0.708 (calibration failure under 7.6% positive rate). MLP_sklearn Apnea F1=0.164 (underfits 383 features). |
| **Key finding** | ElasticNet captured Insomnia's complementary features — +35.9% relative improvement over Exp 3. XGBoost most robust to high-dimensional noise. |

---

### Experiment 5 — PSG Signal-Derived Features
| Property | Value |
|---|---|
| **Notebook** | `project48_psg_features_eval.ipynb` |
| **Input** | `output.csv` (136-col PSG signal CSV, 2,056 subjects) |
| **Preprocessing** | HRV zeros→NaN→median impute→log1p · Drop MeanHR per stage (91–95% zeros) · Drop EEG1_N2_Sigma_mean (duplicate) · Correlation pruning \|r\|>0.95 (28 pairs) · log1p on EEG power features · RobustScaler |
| **MI filter issue** | 30th-percentile threshold resolved to 0.000 on this dataset — all 106 post-pruning features passed. Hard threshold (mi ≥ 0.003) should be used instead. |
| **Runtime** | ~3.5 min (11× faster than Exp 4) |
| **Results** | |
| Apnea | F1=0.183 (collapsed — label is self-reported, PSG signal doesn't encode diagnosis history) |
| Insomnia | F1=0.130 (AUROC=0.494, below random ranking) |
| RLS | F1=0.182 **(PSG wins — PLM_N and PLMI are direct physiological markers)** |
| **Label-feature alignment finding** | PSG features cannot predict self-reported questionnaire labels. Questionnaire labels encode: "were you told you have this disorder?" — which correlates with CPAP use, doctor visits, and snoring complaints, not raw physiology. Exception: RLS, where PLM leg movement counts are a direct physiological signature. |

---

### Experiment 6 — Merged PSG + Questionnaire (BROKEN — do not use results)
| Property | Value |
|---|---|
| **Notebook** | `experiment6_merged.ipynb` |
| **Input** | Inner join: PSG CSV + questionnaire CSV on nsrrid=mesaid → 2,056 rows × ~757 cols |
| **Three root cause failures** | |
| Failure 1 | **MI pre-filter was never ported** — pipeline went directly to stability selection on 693 features |
| Failure 2 | **Wrong stability selection regime** — p/n=0.42 (must be <0.15). ElasticNet kept 95%+ of 693 features — not selection at all. |
| Failure 3 | **No early stopping on PyTorch MLP** — RLS train_loss=0.052 vs val_loss=5.65 at epoch 60. Model trained 40 epochs past divergence. |
| **Results (invalid)** | Apnea=0.784 (held — XGBoost is robust to dimensionality) · Insomnia=0.213 (−45% vs Exp 4) · RLS=0.115 (worst ever) |
| **Status** | Results invalid. Use Experiment 6b. |

---

### Experiment 6b — Merged PSG + Questionnaire (FIXED — pending Kaggle run)
| Property | Value |
|---|---|
| **Notebook** | `experiment6b_fixed.ipynb` |
| **Fixes applied** | |
| Fix 1 | Two-stage feature selection: MI pre-filter (mi ≥ 0.002) first → 693 to ~120–180 features, then stability selection (p/n drops from 0.42 to <0.15) |
| Fix 2 | PyTorch MLP early stopping: patience=10 on validation loss, best-checkpoint restoration |
| Fix 3 | Dropped LinearSVM (consistently fails to converge at 600+ features) |
| **Expected feature count** | ~40–80 per target after two-stage filter |
| **Architecture change** | PyTorch MLP reduced from 512→256→128→64 back to 256→128→64 (deeper model made overfitting worse when features are mostly noise) |
| **Status** | Notebook delivered, not yet run on Kaggle. Results pending. |
| **Expected outcome** | If MI filter + stability selection produce the correct p/n regime, merged features should outperform both Exp 4 and Exp 5 on RLS (PLM signal from PSG + behavioural correlates from questionnaire). Apnea should hold at ~0.78. Insomnia is the most uncertain — depends on how many questionnaire behavioural features survive the MI filter. |

---

### Temporal-Spectral Fusion Model (Experiment — ongoing)
| Property | Value |
|---|---|
| **Notebook** | `temporal-spectral-fusion__9_.ipynb` |
| **Task** | Multi-class sleep staging: W / N1 / N2 / N3 / REM (5 classes) |
| **Input A** | `x_temporal`: raw EEG signal [window_size, 3000] — 30 seconds at 100Hz |
| **Input B** | `x_spectral`: pre-computed spectral features [window_size, 34] |
| **Window size** | 256 epochs per chunk, overlap=0 |
| **Architecture** | See Section 5 |
| **Training** | 150 epochs, Focal Loss (γ=2, per-class alpha weights), AdamW, ReduceLROnPlateau, DataParallel |
| **Status** | Model trains successfully on Kaggle GPU. Explainability pipeline (GradCAM + AttentionRollout + MCDropout) implemented and tested. Full cross-disorder evaluation not yet benchmarked against Experiments 1–6. |

---

### Experiment Results Reference Table

| Experiment | Apnea F1 | Insomnia F1 | RLS F1 | Metric note |
|---|---|---|---|---|
| Exp 1 — RNN | 0.614 | 0.568 | 0.540 | ROC-AUC |
| Exp 2 — Manual | 0.757 | 0.629 | 0.622 | ROC-AUC |
| Exp 3 — Auto L1 | 0.782 | 0.286 | 0.166 | F1 |
| **Exp 4 — ElasticNet** | **0.784** | **0.388** | 0.155 | **F1 — best questionnaire** |
| Exp 5 — PSG signals | 0.183 | 0.130 | **0.182** | F1 — best RLS |
| Exp 6 — Merged (broken) | 0.784 | 0.213 | 0.115 | F1 — invalid |
| Exp 6b — Merged (fixed) | TBD | TBD | TBD | F1 — pending run |

---

## 5. Model Architecture — Temporal-Spectral Fusion

### Overview
Two parallel encoders process temporal and spectral inputs independently, then fuse via bidirectional cross-attention. A transformer context encoder models temporal dependencies across epochs within a window.

### Components

#### Temporal encoder: `AdaptiveAtrousPyramid`
- Input: raw EEG signal `[B, T, 3000]`
- Multi-scale dilated convolutions with rates (1, 2, 4, 8) across branches
- Each branch: Conv1d → BatchNorm → ReLU → SEBlock (squeeze-and-excite channel attention)
- Branches concatenated → projection layer → output `[B, T, embed_dim]`

#### Spectral encoder
- Input: spectral features `[B, T, 34]`
- Linear projection + BatchNorm + ReLU → output `[B, T, embed_dim]`

#### Fusion: `BidirectionalCrossAttnFusion`
- Temporal attends to spectral AND spectral attends to temporal simultaneously
- Gate network: `sigmoid(Linear([h_temp; h_spec]))` → weighted combination
- Alternative: `GatedFusion` (simpler learned gate without cross-attention)

#### Context encoder
- Transformer encoder with positional encoding
- Models dependencies across epochs within the window
- Output: `[B, T, embed_dim]`

#### Output head
- Per-epoch classification: `Linear(embed_dim, 5)` → 5 sleep stages
- Returns: `logits [B, T, 5]`, attention weights, intermediate embeddings

### Loss function: `FocalLoss`
- γ=2.0, per-class alpha weights for class imbalance
- Padding mask (`-100`) correctly excluded from the mean
- Works with DataParallel (buffers registered correctly)

### Dataset: `FusionSleepDataset`
- Sliding-window, window_size=256 epochs, overlap=0
- Returns: `x_temporal [W, 3000]`, `x_spectral [W, 34]`, `y [W]`
- `fusion_collate_fn` handles variable-length windows with zero-padding

---

## 6. Explainability Pipeline

All three components are implemented in `temporal-spectral-fusion__9_.ipynb` cells 18–21 and the Angular app `SleepModelService`.

### AttentionRollout
- **Purpose**: Per-epoch influence scores — "which epochs in the window drove the prediction?"
- **Method**: Monkey-patches each `self_attn.forward` to force `need_weights=True`, bypassing PyTorch's fast path (which returns `None` for weights). Accumulates attention maps across transformer layers.
- **Output**: `influence [B, T]` — normalised influence score per epoch

### EpochGradCAM
- **Purpose**: Within-epoch signal localisation — "which parts of the 30-second EEG waveform did the model focus on?"
- **Method**: Hook on `AdaptiveAtrousPyramid` projection layer. Single forward pass with `retain_grad()`. Gradient flows back to the captured feature map.
- **Output**: `heatmap [3000]` — attention over raw EEG samples, normalised to [0, 1]
- **Visualisation**: Regions above 80th percentile of smoothed heatmap highlighted in red (attention), rest shown in green (background). Top 3 regions annotated as "Pattern 1/2/3 (AI focused here)".

### MCDropoutPredictor
- **Purpose**: Per-epoch uncertainty quantification — "how confident is the model in this epoch's prediction?"
- **Method**: Forces all Dropout layers to `.train()` mode during eval. Runs 30 forward passes. Computes predictive entropy: `H = -Σ p_k log p_k`.
- **Output**: `mean_probs [B, T, C]`, `uncertainty [B, T]` (entropy), `pred_std [B, T, C]`
- **Confidence mapping**: `ratio = entropy / log(5)`. If ratio < 0.45 → High, < 0.72 → Medium, else → Low.
- **Flagging rule**: Epoch flagged for manual review if `prediction ≠ ground_truth` OR `entropy > 75th percentile of recording`.

### Visualisation functions (`explain_single_sample`)
Produces three clinical figures:
1. **Figure 1 — Hypnogram**: Confidence-coloured epoch strips (jet colormap) + ground truth (green, thick) + AI prediction (red, dashed) + agreement % + correct/wrong bar strip
2. **Figure 2 — Review flags**: Epoch timeline coloured by sleep stage, confidence dots per epoch, red borders on disagreements, annotation arrows on top-5 worst (low conf + wrong) epochs
3. **Figure 3 — Signal detail**: 30s raw EEG with GradCAM regions highlighted, epoch header strip, summary card (stage match/mismatch, confidence, time, review alert)

---

## 7. File Map — Inputs and Outputs

### Research notebooks (Kaggle)

| File | Input | Output | Status |
|---|---|---|---|
| `feature_studyv3.py` | `mesa-sleep-dataset-0.8.0.csv` | Feature importance scores, F1 per model | Exp 3 baseline script |
| `project48_feature_study_v3.ipynb` | `mesa-sleep-dataset-0.8.0.csv` | `experiment3_results.csv` | Exp 3 — complete |
| `project48_feature_study_v3_optimised.ipynb` | `mesa-sleep-dataset-0.8.0.csv` | `best_model_summary.png`, `Full_summary.png`, per-target stable feature lists | **Exp 4 — best questionnaire result** |
| `project48_feature_study_v3_full_eval.ipynb` | `mesa-sleep-dataset-0.8.0.csv` | Full eval suite with ROC/PR curves | Exp 4 extended eval |
| `project48_psg_features_eval.ipynb` | `output.csv` | `clinical_fig1_sleep_summary.png`, `clinical_fig2_review_flags.png`, `clinical_fig3_signal_detail.png` | Exp 5 — complete |
| `temporal-spectral-fusion__9_.ipynb` | `index.csv` + `.pt` tensors + spectral `.pt` files | `best_model_fusion.pt`, three clinical figures | Exp 7 (fusion) — training complete |
| `experiment6_merged.ipynb` | `output.csv` + `mesa-sleep-dataset-0.8.0.csv` | `experiment6_merged_summary.csv`, `exp6_three_way_comparison.png` | Exp 6 — broken, use 6b |
| `experiment6b_fixed.ipynb` | `output.csv` + `mesa-sleep-dataset-0.8.0.csv` | `experiment6b_summary.csv`, `exp6b_four_way_comparison.png` | Exp 6b — pending Kaggle run |

### Data files

| File | Description | Location |
|---|---|---|
| `output.csv` | PSG signal-derived features, 2,056 × 136, targets: Insomnia/RLS/apnea | `/mnt/user-data/uploads/output.csv` |
| `mesa-sleep-dataset-0.8.0.csv` | Questionnaire CSV, 2,237 × 628, targets: insmnia5/rstlesslgs5/slpapnea5 | Kaggle input only |
| `apnea_exp6_stable_features.csv` | 663 stable features for Apnea (Exp 6, broken) | `/mnt/user-data/uploads/` |
| `Insomnia_exp6_stable_features.csv` | 660 stable features for Insomnia (Exp 6, broken) | `/mnt/user-data/uploads/` |
| `RLS_exp6_stable_features.csv` | 664 stable features for RLS (Exp 6, broken) | `/mnt/user-data/uploads/` |
| `experiment6_merged_summary.csv` | Exp 6 full model×target results table | `/mnt/user-data/uploads/` |

### Deliverables

| File | Description |
|---|---|
| `project48_experiment_report.docx` | Full Word report covering all 6 experiments |
| `project48_slides_exp3to5.pptx` | 15-slide deck (Experiments 3–5 + summary + conclusions) |
| `sleep_explainability_system.jsx` | Original React prototype of the explainability app |
| `sleep-explainer.zip` | Complete Angular 21 platform source (SomnAI) |

---

## 10. Key Technical Decisions and Constraints

### ML / Research

| Decision | Rationale |
|---|---|
| Use F1 + PR-AUC, never ROC-AUC as primary | At 4–8% positive rate, ROC-AUC is deceptive. A model predicting all-negative scores AUROC≈0.5 but F1=0. |
| ElasticNet l1_ratio=0.5 (equal L1+L2) | L1 alone selects one feature from correlated groups, discarding the rest. L2 keeps them together for complementary feature sets. |
| Stability threshold must satisfy p/n < 0.15 | Meinshausen & Bühlmann stability selection assumes p << n. At p/n > 0.3, ElasticNet keeps hundreds of non-zero coefficients regardless of C — threshold becomes meaningless. |
| MI pre-filter before stability selection | For merged datasets (693 features), the MI filter (mi ≥ 0.002) is mandatory — it brings p/n from 0.42 to ~0.09, putting stability selection in its correct operating regime. |
| pos_weight in BCEWithLogitsLoss | Critical for all targets at 4–8% positive rate. `pos_weight = n_negative / n_positive`. Without it, models collapse to predicting all-negative. |
| RobustScaler over StandardScaler | EEG power features and HRV have extreme outliers (z > 5). RobustScaler uses IQR, making it outlier-resistant. |
| Threshold tuning from OOF probabilities | Default threshold=0.5 is wrong at low positive rates. Tune threshold to maximise F1 from out-of-fold probabilities on training set. |
| MLP_sklearn caps out at ~100 features | Shallow 2-layer sklearn MLP underfits on >100 features. Use MLP_PyTorch (256→128→64→1) with BCEWithLogitsLoss + pos_weight + early stopping for high-dimensional data. |
| LinearSVM unreliable at 600+ features | Fails to converge. Raise max_iter=10000 or drop from high-dimensional experiments. |

### Angular / Platform

| Decision | Rationale |
|---|---|
| Angular signals (not RxJS/BehaviorSubject) | Angular 21 signals are simpler, synchronous, and more readable. `computed()` auto-invalidates downstream when dependencies change — no manual subscription management. |
| `AnalysisStateService` as single shared store | All five explainability tabs read from the same signals. Changing disorder or preset triggers all computed values to recalculate simultaneously. |
| Direct Anthropic API fetch (no proxy) | Simplest for a demo/research platform. In production, proxy through a backend to protect the API key. |
| Mock data + seeded RNG | Allows deterministic, reproducible demo without needing a live backend or real patient data. |
| `[attr.stroke-dasharray]` not `stroke-dasharray` binding | Angular requires `[attr.X]` for SVG attributes that aren't standard DOM properties. String interpolation `{{}}` inside SVG attribute values causes template errors. |
| `color-mix(in srgb, var(--accent) N%, transparent)` | Clean way to derive tinted backgrounds from a single accent variable without defining multiple color variables per theme. |
| No RouterLink in components that don't route | Angular 21 throws NG8113 warning for unused imports — keep imports minimal. |

---

## 11. Known Issues and Pitfalls

### Research

1. **Exp 6 feature lists are invalid** — the 660+ feature lists in the uploaded CSVs represent 95% retention from a broken pipeline. Do not use them as a reference for what features actually matter.

2. **PSG targets vs questionnaire targets** — `Insomnia` (PSG CSV) and `insmnia5` (questionnaire CSV) are the same underlying label (both binary, both from MESA). When merging, drop the questionnaire targets and use PSG targets — they're already clean binary integers with no NaN.

3. **`EEG1_N2_Sigma_mean` is a duplicate** — identical to `EEG1_N2_Sigma_Power_mean`. Drop one before any analysis.

4. **HRV zeros are not true zeros** — `HRV_SDNN_ms=0` means the HRV computation failed, not that the patient had no heart rate variability. Replace with NaN before imputation.

5. **MeanHR per stage columns** — `MeanHR_W_bpm`, `MeanHR_N1_bpm`, etc. have 91–95% zeros. Drop these columns entirely.

6. **MI filter percentile threshold resolves to 0.000 on PSG data** — the 30th-percentile approach used in Experiment 5 was inactive because most features had MI ≈ 0. Use a hard threshold (mi ≥ 0.002 for merged data, mi ≥ 0.003 for PSG-only).

7. **XGBoost inflated AUROC on LinearSVM** — LinearSVM AUROC=0.918 on Apnea does not reflect clinical utility. Its F1=0.708 is the correct performance measure because calibration fails at 7.6% positive rate.

### Platform

1. **`cat >>` corrupts TypeScript class files** — appending after the closing `}` brace puts methods outside the class. Always rewrite files completely rather than appending.

2. **Math.min / Math.max in Angular templates** — Angular templates don't have access to `Math`. Wrap in a computed signal or component method.

3. **SVG attribute bindings** — use `[attr.stroke-dasharray]="expression"` not `stroke-dasharray="{{expression}}"`. The latter causes parse errors for non-DOM SVG attributes.

4. **Typed lambda in Angular templates** — `array.map((n: string) => n[0])` fails in templates. Move to a component method.

5. **Record<string, T> indexing** — TypeScript's `noPropertyAccessFromIndexSignature` flag blocks `obj.property` access on `Record<string, T>`. Use `obj['property']` or create a helper method.

6. **`DisorderId` type location** — it lives in `sleep.models.ts`, not `platform.models.ts`. Easy to import from the wrong file.

---

## 12. Desired Goals and Open Research Gaps

### Immediate next steps (ordered by priority)

1. **Run Experiment 6b on Kaggle** — the fixed two-stage feature selection notebook is ready. Execute and record results. If RLS F1 > 0.182 (Exp 5 best), the complementarity hypothesis is confirmed.

2. **Complete Temporal-Spectral Fusion evaluation** — run the fusion model's full cross-disorder evaluation (Apnea/Insomnia/RLS) using the trained checkpoint. Compare against Experiments 1–6 as the definitive Experiment 7.

3. **Connect real model outputs to SomnAI** — replace `SleepModelService` synthetic inference with API calls to a deployed endpoint serving the actual trained XGBoost (Exp 4) or fusion model. The Angular service layer is already architected for this — one method swap.

4. **Build EDF preprocessing pipeline in-app** — currently users must upload pre-processed `.pt` files from a Kaggle pipeline. A Python backend (FastAPI) that accepts `.edf` and runs preprocessing + inference would make the platform end-to-end.

### Open research gaps

| Gap | Description | Suggested approach |
|---|---|---|
| **RLS physiological label** | Self-reported RLS label is weak. Best RLS F1=0.182. | Create AHI-threshold apnea label (AHI>15) and PLM-threshold RLS label (PLMI>15) from signal data instead of questionnaire. Expected to dramatically improve PSG-feature models. |
| **Raw EMG time series for RLS** | Aggregated PLM statistics (PLM_N, PLMI) capture RLS moderately. Raw windowed leg EMG time series would be far more direct. | Extend temporal encoder to accept raw leg EMG signal alongside EEG. |
| **Multi-task learning** | Insomnia and RLS share sleep architecture correlates. Training separate models ignores this. | Shared encoder, three disorder-specific heads. May improve all three simultaneously, especially RLS. |
| **XGBoost SHAP for clinical explainability** | XGBoost is the best-performing model (Apnea F1=0.784). Apply `shap.TreeExplainer` to get exact SHAP values (not the approximations in `SleepModelService`). | Run SHAP on the saved XGBoost checkpoint and surface values in the Angular app via API. |
| **Cross-dataset generalisation** | All experiments use MESA Sleep Study only. Performance on other PSG datasets (SHHS, CHAT, CFS) is unknown. | Zero-shot evaluation on at least one held-out dataset. |
| **Epoch-level disorder prediction vs recording-level** | Current disorder labels are recording-level (one label per patient). The model predicts per-epoch but the label doesn't reflect epoch-level variation. | Investigate whether per-epoch disorder probability (from the fusion model's per-epoch output) has clinical utility as a temporal disorder burden score. |
| **Personalised lifestyle recommendation system** | The project title includes "Personalised Lifestyle Recommendations" — this is currently unimplemented beyond the AI advisor chat. | Design a structured recommendation engine that maps specific feature deviations (high WASO + low N3) to specific validated lifestyle interventions (sleep restriction therapy, exercise timing, caffeine cutoff) rather than relying solely on LLM generation. |

---

## 13. Conventions and Non-Negotiables

### Research conventions
- **Primary metric is always F1** (+ PR-AUC secondary). Never report ROC-AUC as the headline for imbalanced classification.
- **Always report per-disorder separately** — aggregated accuracy is meaningless given class imbalance.
- **Baselines for comparison**: Exp 4 is the questionnaire baseline (0.784/0.388/0.155). Exp 5 is the PSG baseline (0.183/0.130/0.182). Any new experiment must be compared against both.
- **Threshold tuning**: Always tune classification threshold from OOF probabilities (precision_recall_curve on training fold), not from test set.
- **Stability selection regime**: Always verify p/n ratio after MI pre-filter. Target p/n < 0.15 before running stability selection.
- **Dataset split is at file level** (not epoch/chunk level) to prevent data leakage. Use `train_test_split(indices, test_size=0.2, random_state=42)`.

### Angular conventions
- **Standalone components only** — no NgModules anywhere. Every component declares its own `imports` array.
- **Signals for state** — use `signal()`, `computed()`, `effect()`. No RxJS `BehaviorSubject` or manual subscriptions for new code.
- **No inline styles in templates** — all colours via `[style.color]` or `[style.--custom-property]` binding to component values. No hardcoded hex in templates.
- **Rewrite files, never append** — `cat >>` corrupts TypeScript class structure. Always write complete files.
- **Patient vs clinician separation** — never add EEG viewer links, confidence scores, entropy values, model accuracy, SHAP terminology, or epoch-level technical metrics to any patient-facing template. The patient sees: risk score, feature importance %, counterfactuals in human language, AI advisor chat.
- **Three Claude API agents are distinct**:
  1. Analysis Agent — writes clinical summary from `buildContext()`
  2. Report Agent — writes patient-friendly narrative for PDF/HTML download
  3. Advisor Agent — conversational, maintains full message history, forbidden from technical jargon in system prompt

### Naming
- Disorder IDs: always lowercase `'apnea' | 'insomnia' | 'rls'` (matches PSG CSV column names)
- Questionnaire targets: `insmnia5`, `rstlesslgs5`, `slpapnea5` (note the typo in `insmnia5` — this is the actual column name in the MESA CSV)
- PSG targets: `Insomnia`, `RLS`, `apnea` (mixed case — this is how the output.csv is structured)
- Sleep stages: `W=0, N1=1, N2=2, N3=3, REM=4`
- Stage colours: Wake=#E05C5C · N1=#A78BCA · N2=#5B9BD5 · N3=#4BAE8A · REM=#F0A500
- Confidence colours: High=#34d399 (green) · Medium=#fbbf24 (amber) · Low=#f87171 (red)

---
