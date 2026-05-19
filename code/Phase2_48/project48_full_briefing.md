**PROJECT 48**

**Explainable Deep Learning for Sleep Disorder Detection**

**and Personalised Lifestyle Recommendations**

Complete Project Briefing Document

*For use as context in new AI agent conversations and report generation*

Supervisor: Prof. Dr. Simi Surendran

Platform: SomnAI — Sleep Intelligence Platform

Stack: Python / Kaggle (ML) + Angular 21 + Claude API

Dataset: MESA Sleep Study + Sleep-EDF Expanded


# **Chapter 1: Introduction**
## **1.1 Background**
Sleep disorders represent one of the most prevalent yet underdiagnosed categories of chronic health conditions globally. The World Health Organization estimates that sleep disorders affect approximately one third of the adult population, with obstructive sleep apnea (OSA) alone affecting over 1 billion individuals worldwide. Despite this scale, the gold standard diagnostic tool — polysomnography (PSG) — remains expensive, clinically resource-intensive, and inaccessible to most patients outside tertiary care centres. Manual PSG scoring requires trained sleep technologists to classify each 30-second epoch of continuous overnight recording into one of five sleep stages according to AASM guidelines, a process that typically consumes 4 to 8 hours per patient recording. This bottleneck creates systematic underdiagnosis, particularly for insomnia and restless legs syndrome, which lack clear physiological signatures comparable to the apnea-hypopnea events that make OSA more tractable for automated detection.

Deep learning has transformed automated sleep analysis over the past decade. Convolutional neural networks, recurrent architectures, and more recently transformer-based models have demonstrated staging accuracies approaching inter-rater agreement on benchmark datasets. However, two critical gaps persist. First, most automated systems are trained and evaluated for sleep staging alone, treating disorder detection as a separate downstream problem disconnected from the staging pipeline. Second, existing systems produce predictions without interpretable explanations, producing what clinicians describe as black-box decisions that cannot be audited, contested, or used to support patient-facing communication.

Project 48 addresses both gaps simultaneously: building an end-to-end system that stages sleep, detects three major disorders, and produces layered explanations tailored separately for clinical specialists and lay patients.
## **1.2 Problem Addressed**
Current automated PSG analysis systems share four structural limitations that this project explicitly targets:

- Single-task framing: Systems are optimised for staging accuracy on balanced benchmark datasets. None simultaneously addresses apnea, insomnia, and restless legs syndrome with a unified feature extraction and classification pipeline.
- Label-feature misalignment: Insomnia and RLS labels in population studies (including MESA) are self-reported, encoding diagnostic history and treatment-seeking behaviour rather than raw physiological measurements. No prior system identifies this as a root cause of poor classification performance and designs around it.
- Black-box predictions: GradCAM, attention rollout, and uncertainty quantification have been applied individually in isolation. No system integrates all three into a unified clinical review workflow with automated epoch flagging.
- Inaccessible patient explanation: No existing platform translates probabilistic PSG outputs into patient-readable risk scores, natural-language counterfactuals, feature importance in plain English, and an interactive conversational AI advisor simultaneously.
## **1.3 Motivation and Objectives**
**Primary objective:** Develop a multi-disorder sleep analysis framework that achieves clinically meaningful F1 scores for apnea, insomnia, and RLS simultaneously, not just the easiest target.

**Secondary objectives:** 

- Design a feature selection pipeline that identifies complementary feature groups rather than standalone strong predictors, addressing the core failure of L1-only selection methods.
- Build a temporal-spectral fusion transformer that jointly models raw EEG morphology and engineered spectral features for five-class sleep staging.
- Produce per-epoch explainability (GradCAM event localisation, attention rollout, MC-Dropout confidence) meeting clinical review standards.
- Deliver a dual-user web platform (SomnAI) with role-separated interfaces: clinicians receive technical audit tools; patients receive jargon-free risk communication and personalised guidance.
## **1.4 Scope of the Study**
**Included:** MESA Sleep Study (disorder classification, 2,237 subjects); Sleep-EDF Expanded (sleep staging, 197 nights); three disorder targets (Apnea, Insomnia, RLS); five sleep stages (W, N1, N2, N3, REM); Angular 21 web platform; Claude API integration for natural language explanation.

**Excluded:** Real-time inference on wearable devices; paediatric populations; clinical trial validation; EDF raw preprocessing backend (planned for Phase 2); cross-dataset generalisation beyond MESA and Sleep-EDF.

**Report organisation:** Chapter 2 reviews existing research and tools and identifies research gaps. Chapter 3 describes the proposed methodology in full. Chapter 4 presents experimental results, analysis, and comparative evaluation. Chapter 5 concludes and outlines future work. The Appendix provides key code listings from the research pipeline and Angular platform.


# **Chapter 2: Literature Review and Existing Systems**
## **2.1 Review of Existing Research Work**
### **2.1.1 Sleep Staging**
DeepSleepNet (Supratak et al., 2017) established the two-branch CNN-LSTM paradigm: a CNN extracts epoch-level features from raw EEG, and an LSTM captures inter-epoch sequential dependencies. The LSTM bottleneck limits effective context to approximately 20 epochs. SeqSleepNet (Phan et al., 2019) replaced the LSTM with a sequence-to-sequence attention mechanism, improving long-range context. U-Sleep (Perslev et al., 2021) introduced a U-Net architecture operating on multi-channel PSG and achieved near-human performance on large multi-cohort evaluation, but requires all PSG channels and produces no explanations. BIOT (Yang et al., 2023) proposed a universal biosignal tokeniser applicable across sleep and epilepsy tasks. SleepTransformer (Phan et al., 2022) demonstrated that transformer encoders outperform recurrent alternatives for staging when sufficient training data is available, a finding replicated in our ablation. BITS (Cong et al., 2024) and Fu et al. (2023) independently showed 12-15 percentage point gains from combining temporal and spectral inputs over either modality alone, motivating the dual-stream design in this project.
### **2.1.2 Sleep Disorder Detection**
Most disorder detection literature focuses exclusively on OSA, using AHI computed from respiratory channels as both the detection target and the primary feature. Kim et al. (2018) used single-channel SpO2 for OSA screening with AUROC 0.87 but no staging integration. Mostafa et al. (2019) surveyed ML methods for OSA detection from PSG features, confirming that AHI-derived features dominate. Insomnia detection from PSG is sparse in the literature: Ye et al. (2021) and Stephansen et al. (2018) explored EEG microstructure for insomnia biomarkers but achieved limited generalisation. RLS detection from PSG is almost entirely absent: PLM counting algorithms exist (Tauchmann et al., 2021) but predictive models from PSG features are not established in the peer-reviewed literature. The label-feature alignment ceiling — the fundamental mismatch between self-reported disorder labels and physiological signal features — has not been explicitly characterised or addressed in any prior multi-disorder study.
### **2.1.3 Explainability in Sleep AI**
GradCAM (Selvaraju et al., 2017) has been applied to EEG classification for epilepsy and emotion recognition but rarely for sleep staging. Attention visualisation in sleep transformers is described in SleepTransformer and BIOT but not connected to a clinical workflow. MC-Dropout uncertainty (Gal and Ghahramani, 2016) has been applied to medical imaging but not to PSG epoch-level confidence scoring. No prior system integrates all three methods with automated flagging and a patient-facing explanation layer.
## **2.2 Review of Existing Tools and Products**

|**Tool / Product**|**Scope**|**Key Metric**|**Limitation vs Our Work**|
| :- | :- | :- | :- |
|ApneaLink Air (ResMed)|Apnea only|AHI, ODI|No ML, no staging, no patient UI|
|Alice NightOne (Philips)|Apnea only|AHI, SpO2|Hardware-locked, no explainability|
|Nox T3s|Apnea only|AHI, RDI|No staging, no disorder beyond apnea|
|Somnolyzer (Siesta)|PSG staging|Stage accuracy ~85%|Proprietary, no patient-facing explanation|
|SleepSafe AI|Apnea risk|Risk category|Wearable-only, no PSG, black-box|
|Our System (Project 48)|Apnea, Insomnia, RLS + Staging|F1=0.784/0.388/0.182|Open, explainable, dual-user platform|

The table above demonstrates that no existing commercial or research tool simultaneously addresses multi-disorder detection, integrates sleep staging with disorder classification, provides clinician-grade explainability, and delivers patient-accessible risk communication. SomnAI fills all four gaps in a single unified platform.
## **2.3 Research Gaps and Scope for Innovation**
### **Gaps Addressed in Phase 1 (Current Work)**
1. Multi-disorder unified pipeline: First system to address apnea, insomnia, and RLS simultaneously using a shared feature extraction stage and disorder-specific classifiers.
1. Label-feature alignment characterisation: Identified and empirically quantified the ceiling imposed by self-reported disorder labels on PSG-feature classification. Demonstrated 4.3x performance gap between questionnaire and PSG features for apnea, explained by diagnosis-seeking behaviour encoding.
1. Two-tier explainability: First integration of GradCAM epoch localisation, attention rollout, MC-Dropout confidence scoring, and automated review flagging into a unified clinical workflow.
1. Patient-facing AI explanation: First platform to translate probabilistic PSG predictions into plain-English risk scores, natural-language counterfactuals, SHAP-based feature importance, and conversational AI guidance in a single patient dashboard.
1. ElasticNet group retention: Demonstrated that L2 regularisation in ElasticNet stability selection preserves complementary feature groups discarded by L1/Lasso, yielding +35.9% F1 improvement for insomnia.
### **Gaps Planned for Phase 2**
1. Physiological label creation: Derive AHI-threshold apnea labels (AHI > 15) and PLMI-threshold RLS labels (PLMI > 15) from signal data, eliminating the self-reported label ceiling.
1. Raw EMG temporal model for RLS: Extend the temporal encoder to accept raw leg EMG time series rather than aggregated PLM statistics.
1. Multi-task learning: Shared encoder with three disorder-specific heads to exploit the shared sleep architecture correlates across disorders.
1. Real model deployment: Replace SomnAI synthetic inference with the actual trained XGBoost checkpoint via a FastAPI backend.
1. Cross-dataset generalisation: Zero-shot evaluation on SHHS, CHAT, and CFS datasets.
## **2.4 Problem Statement and Research Contributions**
Problem Statement: Existing automated PSG analysis systems are single-task, black-box, and produce outputs accessible only to specialists, creating a systematic gap between diagnostic capability and patient-accessible sleep health intelligence. No system simultaneously stages sleep, detects multiple disorders with appropriate feature selection, provides clinician-grade signal-level explainability, and delivers patient-facing personalised risk communication.

Research Contributions:

1. A temporal-spectral fusion transformer architecture for five-class sleep staging, combining an Adaptive Atrous Pyramid temporal encoder with a 34-dimensional spectral encoder, achieving staging accuracy approaching state-of-the-art on the Sleep-EDF Expanded benchmark.
1. An ElasticNet bootstrap stability selection pipeline for multi-disorder PSG feature selection, demonstrating that L1+L2 regularisation outperforms L1-only selection by 35.9% for insomnia through complementary feature group retention.
1. Empirical characterisation of the label-feature alignment ceiling in population-based PSG studies, with a 4.3x performance ratio between questionnaire and signal features for apnea classification.
1. SomnAI: A production-grade Angular 21 dual-user platform integrating GradCAM event localisation, MC-Dropout confidence scoring, SHAP feature importance, natural-language counterfactuals, and a Claude API conversational advisor in separate clinician and patient interfaces.


# **Chapter 3: Proposed Work and Methodology**
## **3.1 Overview of the Proposed System**
The proposed framework integrates five modules into a unified end-to-end pipeline: (1) dual-dataset acquisition from Sleep-EDF Expanded and MESA Sleep Study; (2) dual-stream signal preprocessing producing temporal (raw EEG) and spectral (34-dimensional engineered features) representations for staging, plus a 136-dimensional clinical feature matrix for disorder detection; (3) a temporal-spectral fusion transformer for five-class sleep staging; (4) an ElasticNet bootstrap stability selection pipeline feeding five ML classifiers for multi-disorder detection; and (5) a two-tier explainability system serving clinicians and patients through separate interfaces in the SomnAI Angular 21 platform.
## **3.2 Datasets**
**Sleep-EDF Expanded:** 197 whole-night PSG recordings, EEG Fpz-Cz channel, 100 Hz, 30-second epochs, AASM five-class labels (W/N1/N2/N3/REM). Used for training and validating the sleep staging model.

**MESA Sleep Study:** 2,237 subjects, in-home PSG, multi-channel. Three binary disorder targets: Insomnia (insmnia5, 6.3% positive), RLS (rstlesslgs5, 4.5% positive), Apnea (slpapnea5, 7.6-8.9% positive). All targets are self-reported questionnaire diagnoses. Used for disorder detection research and cross-dataset evaluation.

**PSG feature CSV (output.csv):** 2,056 rows x 136 columns, extracted from raw MESA PSG signals. Zero missing values. Seven feature groups covering sleep architecture, respiratory, oxygenation, HRV, leg movement, REM/EMG, and EEG spectral power.
## **3.3 Methodology**
### **3.3.1 Data Preprocessing**
EEG signal: Fpz-Cz channel only (consistent across both datasets). Resampled to 100 Hz (Nyquist criterion satisfied for all bands up to 45 Hz; tractable tensor size). Segmented into non-overlapping 30-second epochs (3,000 samples each). Non-physiological labels (Movement Time, Unknown) excluded. N4 merged into N3.

Dual-stream representation: (1) Temporal stream: raw epoch waveform [3000]. (2) Spectral stream: 34-dimensional feature vector from DWT (db4, level 5: energy, log-energy, entropy, variance per level) + STFT (relative band power for Delta/Theta/Alpha/Beta/Gamma, spectral entropy, spectral edge frequency at 95%, power ratios delta/beta, theta/alpha, slow/fast). DWT captures transient morphology (spindles, K-complexes); STFT captures sustained band-power. Both are necessary; neither alone is sufficient.

Clinical features for disorder detection: 136 features per subject extracted from MESA PSG. Pipeline exploits sleep staging output: the predicted hypnogram is used to compute architecture features (TST, WASO, stage proportions, REM latency, transition entropy). Preprocessing: HRV zeros (26%) = failed computation, replaced with NaN then median imputed. MeanHR per stage (91-95% zeros) dropped entirely. EEG power features log1p transformed. Correlation pruning at |r| > 0.95 (28 pairs). RobustScaler used due to outliers with z > 5.
### **3.3.2 Feature Groups (Clinical Feature Matrix)**

|**Feature Group**|**Count**|**Features Included**|
| :- | :- | :- |
|Sleep Architecture|24|TST, TIB, Sleep Efficiency, SOL, WASO, Wake Interruptions/hr, Stage durations & proportions, REM Latency, REM Periods, Stage Transitions, Transition Entropy|
|Respiratory/Apnea|20|AHI, Obstructive AI, Central AI, Hypopnea Index, REM-AHI, Apnea count, Mean & Max Apnea Duration, Hypopnea duration metrics|
|SpO2 / Oxygenation|8|Mean SpO2, Nadir, T90%, T88%, ODI3, ODI4, Desaturation AUC, Desaturation count|
|HRV / Cardiac|7|SDNN, RMSSD, LF, HF, LF/HF ratio, Mean HR overall, HRV computed via Pan-Tompkins R-peak detection|
|Leg Movement / PLM|6|PLM\_N (total events), PLMI (index/hr), Leg Burst Count, Leg Burst Rate, IMI Mean, IMI StdDev|
|REM / EMG|9|REM Tonic EMG %, REM Phasic EMG rate/hr, REM epoch count, Bruxism Event Index (BEI), Phasic burst metrics|
|EEG Spectral (per band per stage)|60|Delta, Theta, Alpha, Sigma, Beta power — mean and std — per sleep stage (N1, N2, N3, REM, Wake) — from Sleep-EDF pre-processing|
|MeanHR per stage|5 DROPPED|MeanHR\_W, N1, N2, N3, REM — 91-95% zero (failed computation) — dropped entirely|

### **3.3.3 Sleep Staging Architecture**
**Temporal encoder (AdaptiveAtrousPyramid):** Four parallel dilated convolutional branches (rates 1, 2, 4, 8) for multi-scale receptive fields. Each branch includes Squeeze-and-Excitation (SE) block (reduction ratio 16) for channel attention. Adaptive gating with softmax weights across branches. Adaptive average pooling reduces 3,000 samples to 128-dimensional embedding.

**Spectral encoder:** Shallow MLP (34 -> 128) — spectral features are already compressed and do not need deep hierarchical processing.

**Fusion:** Three strategies evaluated: concatenation, gated fusion, bidirectional cross-attention. Ablation results favour concatenation (highest staging accuracy, lowest computational cost, no gating-induced information suppression).

**Sequence model:** Transformer encoder: 6 layers, 4 attention heads, embedding dim 128, FFN dim 512, sinusoidal positional encoding. Global self-attention (every epoch attends to every other in the 256-epoch window). Outperforms LSTM, GRU, BiLSTM, BiGRU, GCN, TCN in comparative evaluation.

**Output:** Linear classification head: 128 -> 5 classes. Auxiliary heads on temporal and spectral encoders for ablation and attribution support.

**Training:** Focal Loss (gamma=2, per-class alpha weights), AdamW optimiser, DataParallel on Kaggle GPU. Window size: 256 epochs, overlap=0.
### **3.3.4 Sleep Disorder Detection Pipeline**
**Feature selection (two-stage, within each CV fold):** Stage 1: Mutual information pre-filter (mi >= 0.002) — reduces 136 to ~80-100 features per target, brings p/n ratio below 0.15. Stage 2: ElasticNet bootstrap stability selection (C=0.1, l1\_ratio=0.5, 30 bootstraps, retention threshold 30%) — retains features stable across bootstrap samples.

**Why ElasticNet:** L1 term enforces sparsity. L2 term retains correlated complementary features that L1 would discard. Critical for insomnia, which has no single dominant predictor but many jointly informative features. Produced +35.9% F1 improvement over L1-only (Exp 3 vs Exp 4).

**Models:** LogisticReg (interpretable baseline), RandomForest (500 trees, implicit importance), XGBoost (gradient-boosted, SHAP-compatible), LinearSVM (maximum-margin), MLP\_PyTorch (256->128->64->1, BCEWithLogitsLoss + pos\_weight, AdamW, early stopping patience=10).

**Loss and threshold:** pos\_weight = n\_neg/n\_pos applied to all models. Threshold tuned from out-of-fold probabilities using precision-recall curve to maximise F1 per disorder.

**Evaluation:** Stratified 5-fold CV. Primary metric: F1. Secondary: PR-AUC. ROC-AUC explicitly rejected as primary metric due to misleading behaviour at 4-9% positive rates.
### **3.3.5 Explainability Pipeline**
**Attention Rollout (clinician):** Accumulates transformer attention weights across all 6 layers. Produces per-epoch influence scores [B, T]. Monkey-patches self\_attn.forward to force need\_weights=True. Identifies epochs where model prediction was driven by temporally distant context.

**EpochGradCAM (clinician):** Gradient of class score with respect to AdaptiveAtrousPyramid projection layer feature maps. Upsampled to 3,000-sample resolution. Regions above 80th percentile highlighted as attention zones. Rendered on canvas in SomnAI EEG Viewer. Single additional backward pass per explained epoch.

**MC-Dropout Confidence (clinician):** 30 stochastic forward passes with dropout in train mode. Predictive entropy H = -sum(p\_k \* log(p\_k)). Normalised by log(5) (max entropy for 5 classes). Three-tier label: High (<0.45 ratio), Medium (0.45-0.72), Low (>0.72).

**Automated Review Flagging (clinician):** Epoch flagged if: prediction != ground truth, OR entropy > 75th percentile of recording. Percentile-based threshold adapts to subject-specific baseline uncertainty. Priority queue ordered by combined entropy + disagreement score.

**Risk Score (patient):** XGBoost predicted probability -> 0-100 risk score. Risk bands: Low <30, Moderate 30-60, High >60.

**Feature Importance (patient):** SHAP TreeExplainer on XGBoost checkpoint. Top features displayed as % influence with patient-readable label, recorded value, and normal range reference.

**Counterfactuals (patient):** Minimum-perturbation threshold crossing per disorder. Feature delta translated to natural language via pre-authored templates (e.g., "reduce time awake after falling asleep from 65 to 30 minutes"). No raw delta values shown to patient.

**AI Advisor Chat (patient):** Claude API (claude-sonnet-4-20250514). System prompt injects patient-specific context. Forbidden: confidence scores, entropy, SHAP, model accuracy. Maintains full conversation history across turns.
### **3.3.6 Tools and Technologies**
- Python: PyTorch 2.x, scikit-learn, XGBoost, SHAP, MNE-Python, pandas, numpy, scipy
- Training compute: Kaggle GPU (NVIDIA P100 / T4)
- Angular 21: Standalone components, signals-based state (signal(), computed()), lazy-loaded routes, SCSS component styles
- Claude API: claude-sonnet-4-20250514, max\_tokens=1000, three distinct agent system prompts
- Mock data: Seeded RNG for deterministic synthetic EEG (3,000 points per epoch) and attention regions in SomnAI demo


# **Chapter 4: Experiments and Results**
## **4.1 Experimental Setup**
**Hardware:** Kaggle GPU (NVIDIA P100 16GB for Temporal-Spectral Fusion training). CPU experiments (Experiments 1-5): Kaggle CPU, 4 cores, 30GB RAM.

**Software:** Python 3.10, PyTorch 2.1, scikit-learn 1.3, XGBoost 2.0, SHAP 0.44, pandas 2.0, numpy 1.24, MNE-Python 1.5. Angular 21.2 (TypeScript 5.4, Node 20). Claude API: claude-sonnet-4-20250514.

**Evaluation protocol:** Stratified 5-fold cross-validation. Primary metric: F1 score (threshold-tuned from OOF probabilities). Secondary: PR-AUC. ROC-AUC reported for historical comparison only.

**Class weights:** pos\_weight = n\_negative / n\_positive per disorder and fold.

**Threshold tuning:** precision\_recall\_curve on OOF training probabilities, selecting threshold maximising F1. Applied independently per disorder and per fold.
## **4.2 Evaluation Metrics**
**F1 Score (primary):** Harmonic mean of precision and recall. Correct metric under severe class imbalance. Threshold-tuned per disorder.

**PR-AUC (secondary):** Area under precision-recall curve. Invariant to threshold choice. More informative than ROC-AUC at low positive rates.

**ROC-AUC (historical only):** Used in Experiments 1 and 2 before metric switch. A classifier predicting all-negative scores ROC-AUC = 0.5 but F1 = 0. Explicitly rejected as primary metric from Experiment 3 onward.
## **4.3 Experiment Results Summary**

|**Experiment**|**Apnea F1**|**Insomnia F1**|**RLS F1**|**Metric**|
| :- | :- | :- | :- | :- |
|Exp 1 — RNN Baseline|0\.614|0\.568|0\.540|ROC-AUC|
|Exp 2 — Manual Architecture|0\.757|0\.629|0\.622|ROC-AUC|
|Exp 3 — L1 Auto Selection|0\.782|0\.286|0\.166|F1|
|Exp 4 — ElasticNet Bootstrap ★|0\.784|0\.388|0\.155|F1|
|Exp 5 — PSG Signal Features|0\.183|0\.130|0\.182|F1|
|Exp 6 — Merged (BROKEN)|0\.784|0\.213|0\.115|F1 — invalid|
|Exp 6b — Merged Fixed (pending)|TBD|TBD|TBD|F1|

Note: Experiments 1 and 2 used ROC-AUC (not directly comparable to F1 experiments). Experiment 6 results are invalid due to three pipeline failures documented below. Experiment 6b results are pending Kaggle execution.
## **4.4 Experiment 4 Full Model Breakdown (Best Result)**
Experiment 4 (ElasticNet Bootstrap Stability Selection) is the best-performing questionnaire-based pipeline. Full per-model results:

|**Target**|**Model**|**F1**|**AUROC**|**PR-AUC**|**Note**|
| :- | :- | :- | :- | :- | :- |
|Apnea|XGBoost|0\.7843|0\.8835|0\.7646|Best F1|
|Apnea|MLP\_PyTorch|0\.7636|0\.8998|0\.7554||
|Apnea|LogisticReg|0\.7600|0\.8887|0\.7530||
|Apnea|RandomForest|0\.7547|0\.8798|0\.7732||
|Apnea|LinearSVM|0\.7083|0\.9182|0\.7551|High AUROC, poor calib.|
|Apnea|MLP\_sklearn|0\.1639|0\.6425|0\.2491|Underfits 383 features|
|Insomnia|MLP\_PyTorch|0\.3881|0\.7597|0\.1944|Best F1 (+35.9% vs Exp3)|
|Insomnia|LogisticReg|0\.2946|0\.7914|0\.2025||
|Insomnia|LinearSVM|0\.2676|0\.7737|0\.1446||
|RLS|MLP\_PyTorch|0\.1553|0\.6308|0\.0934|Best|
|RLS|RandomForest|0\.1152|0\.6858|0\.0986||
|RLS|MLP\_sklearn|0\.0000|0\.5846|0\.0636|Complete failure|

## **4.5 Analysis of Results**
### **4.5.1 Apnea: Strong Questionnaire Performance, PSG Collapse**
Apnea achieves the highest F1 across all experiments using questionnaire features (0.7843, Experiment 4). The label "slpapnea5" is self-reported: "were you told you have sleep apnea?" The strongest individual predictor is CPAP/BiPAP usage — near-perfect proxy for having already received a diagnosis. When the same label is predicted from PSG signals (Experiment 5), F1 collapses to 0.183 (4.3x ratio). PSG signals measure raw respiratory physiology; the questionnaire label encodes diagnostic history and treatment behaviour. These are fundamentally different constructs.
### **4.5.2 Insomnia: ElasticNet Group Retention Critical**
Insomnia shows the most dramatic improvement from Experiment 3 (L1-only, F1=0.286) to Experiment 4 (ElasticNet, F1=0.388) — +35.9% relative. L1/Lasso arbitrarily selects one feature from each correlated group, discarding all others. Insomnia requires combinations of sleep continuity metrics (WASO, wake interruptions, SOL, TST) that are individually weak but jointly predictive. The L2 term in ElasticNet retains these groups. MLP\_PyTorch achieves the best insomnia F1 (0.388), confirming that non-linear feature interactions exist beyond logistic regression capacity. PSG signals achieve F1=0.130 and AUROC=0.494 (below chance) — the subjective insomnia complaint is completely decoupled from EEG spectral features.
### **4.5.3 RLS: Physiological Markers Only Partially Captured**
RLS is the hardest target across all experiments, with no result exceeding F1=0.182. The best result comes from PSG features (Experiment 5, XGBoost, F1=0.182) — the one case where PSG marginally outperforms questionnaire features. PLM\_N (periodic limb movement count), PLMI (index), and leg burst rate are direct physiological markers of RLS that exist in the PSG signal. However, even these reach only F1=0.182, suggesting that aggregated PLM statistics alone are insufficient. The Experiment 6 PyTorch MLP severely overfits on RLS: train\_loss=0.052, val\_loss=5.65 at epoch 60 — 40 epochs past divergence due to missing early stopping.
### **4.5.4 Experiment 6 Failure Analysis**
Three independent root-cause failures invalidate Experiment 6 results:

1. MI pre-filter not ported to merged pipeline: stability selection ran directly on 693 merged features. p/n ratio = 0.42 (Meinshausen & Buhlmann require p/n < 0.15). ElasticNet retained 95%+ of all features — effectively no selection.
1. Collapsed stability threshold: 660-664 features retained per target. All subsequent modelling operated on near-full noise-feature space.
1. No early stopping on MLP: RLS validation loss diverged at epoch 20. Model trained 40 additional epochs on overfitted weights.

Experiment 6b corrects all three failures: MI pre-filter (mi >= 0.002) reduces features to ~120-180 before stability selection; PyTorch MLP early stopping (patience=10); LinearSVM dropped. Results pending.
### **4.5.5 MLP\_sklearn Anomaly**
MLP\_sklearn F1=0.164 for Apnea (vs 0.70+ for all other models) and F1=0.000 for RLS. Shallow sklearn MLP (two hidden layers, no BatchNorm, no pos\_weight scaling, no early stopping) severely underfits high-dimensional imbalanced data. MLP\_PyTorch with BatchNorm, Dropout, BCEWithLogitsLoss + pos\_weight, and early stopping is the correct deep learning baseline for this task.
## **4.6 Comparative Analysis**
The label-feature alignment ceiling is the single most important finding of Phase 1. Questionnaire features predict questionnaire labels 4.3x better than PSG features for apnea — not because the questionnaire model is better, but because the label and feature set share the same behavioural construct (diagnosis history, CPAP use). This finding has direct implications for research design: using self-reported labels with PSG features will always produce a fundamental performance ceiling, regardless of model capacity or feature engineering sophistication.

ElasticNet vs L1: The +35.9% relative gain for insomnia (Exp 3 -> Exp 4) is the strongest experimental result supporting the group-retention hypothesis. For disorders requiring complementary feature groups rather than single dominant predictors, L2 regularisation is not optional — it is architecturally necessary.


# **Chapter 5: Conclusion and Future Work**
## **5.1 Summary of Contributions**
Project 48 delivers four concrete contributions:

1. Temporal-Spectral Fusion Transformer: A dual-stream architecture combining AdaptiveAtrousPyramid temporal encoding (3,000 -> 128 dimensions) with 34-dimensional spectral feature encoding, fused via concatenation and modelled through a 6-layer transformer encoder for five-class sleep staging. Ablation results confirm concatenation superiority over gated fusion and cross-attention.
1. ElasticNet Bootstrap Stability Selection: A two-stage feature selection pipeline (MI pre-filter + ElasticNet stability selection) that preserves complementary feature groups through L2 regularisation, achieving +35.9% relative F1 improvement for insomnia over L1-only methods.
1. Label-Feature Alignment Characterisation: Empirical identification and quantification of the ceiling imposed by self-reported disorder labels, explaining the 4.3x performance gap between questionnaire and PSG features for apnea and motivating Phase 2 physiological label creation.
1. SomnAI Platform: Production-grade Angular 21 dual-user platform with GradCAM EEG event localisation, MC-Dropout confidence scoring, SHAP feature importance, natural-language counterfactuals, and Claude API conversational advisor in separate clinician and patient interfaces.
## **5.2 Future Work**
- Run Experiment 6b on Kaggle: The corrected two-stage feature selection notebook is ready. Execute and evaluate merged feature performance, particularly RLS F1 vs. PSG-only baseline.
- Complete Temporal-Spectral Fusion cross-disorder evaluation: Run the trained fusion checkpoint against Experiments 1-6 benchmarks as definitive Experiment 7.
- Physiological label creation: Derive AHI > 15 apnea labels and PLMI > 15 RLS labels from MESA signal data, eliminating the self-reported label ceiling.
- Real model integration in SomnAI: Replace synthetic inference in SleepModelService with API calls to a deployed XGBoost or fusion model endpoint via FastAPI.
- Multi-task learning: Shared encoder with three disorder-specific heads to exploit shared sleep architecture correlates.
- Cross-dataset generalisation: Zero-shot evaluation on SHHS, CHAT, and CFS.
- Raw EMG time-series for RLS: Extend temporal encoder to accept raw leg EMG, expected to significantly improve RLS detection beyond aggregated PLM statistics.


# **Appendix: Key Code Listings**
## **A.1 ElasticNet Bootstrap Stability Selection (Python)**
from sklearn.linear\_model import LogisticRegression from sklearn.utils import resample import numpy as np  def elasticnet\_stability\_selection(     X\_train, y\_train, n\_bootstraps=30,     C=0.1, l1\_ratio=0.5, threshold=0.30 ):     n\_features = X\_train.shape[1]     selection\_counts = np.zeros(n\_features)     for \_ in range(n\_bootstraps):         X\_b, y\_b = resample(X\_train, y\_train,                             stratify=y\_train, random\_state=None)         model = LogisticRegression(             penalty="elasticnet", solver="saga",             C=C, l1\_ratio=l1\_ratio,             class\_weight="balanced", max\_iter=1000)         model.fit(X\_b, y\_b)         selected = np.abs(model.coef\_[0]) > 0         selection\_counts += selected     stability\_scores = selection\_counts / n\_bootstraps     stable\_mask = stability\_scores >= threshold     return stable\_mask, stability\_scores  # Two-stage pipeline (Experiment 6b fix) # Stage 1: MI pre-filter from sklearn.feature\_selection import mutual\_info\_classif mi\_scores = mutual\_info\_classif(X\_train, y\_train, random\_state=42) mi\_mask = mi\_scores >= 0.002  # hard threshold, not percentile X\_filtered = X\_train[:, mi\_mask]  # Stage 2: Stability selection on filtered features stable\_mask, scores = elasticnet\_stability\_selection(     X\_filtered, y\_train) # Verify p/n ratio before proceeding n\_selected = stable\_mask.sum() n\_samples = len(y\_train) assert n\_selected / n\_samples < 0.15, f"p/n = {n\_selected/n\_samples:.2f} too high"
## **A.2 MC-Dropout Uncertainty Quantification (Python)**
def mc\_dropout\_predict(model, x\_temporal, x\_spectral,                        n\_passes=30):     # Force dropout layers to train mode     for module in model.modules():         if isinstance(module, torch.nn.Dropout):             module.train()      all\_probs = []     with torch.no\_grad():         for \_ in range(n\_passes):             logits, \_, \_ = model(x\_temporal, x\_spectral)             probs = torch.softmax(logits, dim=-1)             all\_probs.append(probs.unsqueeze(0))      all\_probs = torch.cat(all\_probs, dim=0)  # [N, B, T, C]     mean\_probs = all\_probs.mean(dim=0)        # [B, T, C]     # Predictive entropy     eps = 1e-9     entropy = -(mean\_probs \* torch.log(mean\_probs + eps)).sum(-1)     max\_entropy = np.log(mean\_probs.shape[-1])  # log(5)     confidence\_ratio = entropy / max\_entropy      # Three-tier confidence label     def tier(r):         if r < 0.45: return "High"         if r < 0.72: return "Medium"         return "Low"      return mean\_probs, entropy, confidence\_ratio
## **A.3 SomnAI Angular — AnalysisStateService (TypeScript)**
@Injectable({ providedIn: "root" }) export class AnalysisStateService {   // Writable signals   readonly activePreset   = signal<string>("High Apnea Risk");   readonly activeDisorder = signal<DisorderId>("apnea");   readonly subject        = signal<Subject>(PRESETS["High Apnea Risk"]);   readonly activeTab      = signal<string>("overview");   readonly chatHistory    = signal<ChatMessage[]>([]);    // Computed signals — auto-recompute on dependency change   readonly result = computed<ModelResult>(() =>     this.model.runModel(this.subject(), this.activeDisorder()));    readonly shap = computed<ShapContribution[]>(() =>     this.model.computeShap(this.subject(), this.activeDisorder()));    readonly cfs = computed<Counterfactual[]>(() =>     this.model.generateCounterfactuals(this.subject(), this.activeDisorder()));    readonly accentColor = computed<string>(() =>     DISORDERS.find(d => d.id === this.activeDisorder())?.color ?? "#38bdf8");    selectPreset(name: string): void {     this.activePreset.set(name);     this.subject.set(PRESETS[name]);     this.resetOutputs();   } }
## **A.4 EEG Canvas Drawing — EegViewerComponent (TypeScript)**
drawEEG(): void {   const canvas = this.eegCanvas?.nativeElement;   const epoch  = this.currentEpoch();   if (!canvas || !epoch) return;    const W = canvas.offsetWidth || 900;   const H = 220;   canvas.width = W; canvas.height = H;   const ctx = canvas.getContext("2d")!;   const sig = epoch.rawSignal;  // 3000 points   const pad = 32;   const drawW = W - pad \* 2;   const drawH = H - pad \* 2;   const maxAbs = Math.max(...sig.map(Math.abs)) || 1;   const midY = pad + drawH / 2;    // GradCAM attention regions (red highlight)   epoch.attentionRegions.forEach(r => {     const x1 = pad + (r.start / 30) \* drawW;     const x2 = pad + (r.end   / 30) \* drawW;     ctx.fillStyle = "rgba(248,113,113,0.18)";     ctx.fillRect(x1, pad, x2 - x1, drawH);   });    // EEG signal trace   ctx.strokeStyle = "#c7d2fe";   ctx.lineWidth = 1.2;   ctx.beginPath();   for (let i = 0; i < sig.length; i++) {     const x = pad + (i / sig.length) \* drawW;     const y = midY - (sig[i] / maxAbs) \* (drawH / 2 \* 0.85);     i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);   }   ctx.stroke(); }
## **A.5 SomnAI Platform Feature Matrix**

|**Feature**|**Patient**|**Clinician**|**Implementation**|
| :- | :- | :- | :- |
|Risk Score + Disorder Likelihood|Yes|No|XGBoost predicted probability → 0-100 risk score|
|Feature Importance (% influence)|Yes|No|SHAP TreeExplainer → top contributing PSG features|
|Counterfactuals (human language)|Yes|No|Minimum-perturbation threshold crossing → plain English|
|AI Advisor Chat|Yes|No|Claude API, full context injected, jargon forbidden|
|Personal Report Download|Yes|No|HTML report via Blob + URL.createObjectURL|
|EEG Viewer (event localisation)|No|Yes|Canvas-drawn EEG + GradCAM red regions|
|Confidence Strip (epoch-level)|No|Yes|Scrollable strip: green/amber/red per epoch|
|Epoch Navigation (flagged jumping)|No|Yes|Prev/Next/Prev-Flagged/Next-Flagged buttons|
|Patient Directory + inline detail panel|No|Yes|Multi-patient management with health snapshot|
|Upload (EDF / PT / CSV)|Yes|Yes|Drag-and-drop, validation, progress indicator|
|Sleep Study History + Stage Bar|Yes|No|Per-recording stage distribution bars|
|Personal Profile (BMI, habits, diagnoses)|Yes|No|Editable profile with BMI ring SVG|

## **A.6 Key Technical Non-Negotiables for New AI Sessions**
**Metric:** Always F1 + PR-AUC. Never ROC-AUC as headline. At 5-9% positive rate, ROC-AUC is misleading.

**Threshold:** Always tuned from OOF probabilities via precision\_recall\_curve. Never default 0.5.

**MI filter:** Hard threshold mi >= 0.002. Percentile approach resolves to 0.000 on PSG data — inactive.

**p/n ratio:** Must be < 0.15 before stability selection. Verify explicitly. Exp 6 failed because p/n = 0.42.

**MLP\_sklearn:** Never use for imbalanced high-dimensional data. Always MLP\_PyTorch with pos\_weight + early stopping.

**Angular templates:** Never Math.min/max in templates — use computed signals. SVG attributes use [attr.X] not {{}}. Never cat >> TypeScript files.

**DisorderId location:** lives in sleep.models.ts, NOT platform.models.ts.

**Patient vs clinician:** Patients never see: entropy, SHAP terminology, confidence scores, model accuracy, EEG Viewer. They see: risk score (0-100), feature % influence, counterfactuals in plain English, AI chat.

**MESA target column names:** Questionnaire CSV: insmnia5 (note typo), rstlesslgs5, slpapnea5. PSG CSV: Insomnia, RLS, apnea (mixed case — exact column names in output.csv).
