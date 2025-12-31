# Overfitting Fixes and ML Classifier Comparison

## Problem Identified

The neural network model was severely overfitting:
- Training F1: 0.4881 (epoch 32)
- Validation F1: 0.0417 (epoch 32)
- Model predicting almost all positives for insomnia, all negatives for restless leg
- Poor generalization to validation/test sets

## Neural Network Fixes Applied

### 1. **Reduced Model Complexity**
- **Embedding dimension**: 32 → 16
- **RNN hidden dimension**: 128 → 64
- **Combined hidden dimension**: 256 → 128
- **Simplified classifier**: Removed extra hidden layer

### 2. **Increased Regularization**
- **Dropout**: 0.3 → 0.5 (default)
- **Weight decay**: 1e-5 → 1e-3
- **Added LayerNorm**: Better than BatchNorm for variable batch sizes

### 3. **Architecture Changes**
- Simplified prediction head (removed extra layer)
- Added LayerNorm for normalization
- Maintained bidirectional RNN but with smaller dimensions

## Recommended Training Command (Fixed Model)

```bash
python code/train_comorbidity_classifier.py \
    --csv_path mesa_final.csv \
    --output_dir checkpoints/comorbidity_fixed \
    --use_stratified_split \
    --use_weighted_sampling \
    --weight_method per_class \
    --loss_type focal \
    --focal_gamma 2.0 \
    --class_weight_method aggressive \
    --optimize_thresholds \
    --early_stop_metric macro_f1 \
    --early_stop_patience 10 \
    --epochs 50 \
    --batch_size 32 \
    --dropout 0.5 \
    --weight_decay 1e-3 \
    --embedding_dim 16 \
    --rnn_hidden_dim 64
```

## Traditional ML Classifiers

Added support for comparing with traditional ML methods:

### Training ML Classifiers

```bash
python code/train_ml_classifiers.py \
    --csv_path mesa_final.csv \
    --output_dir checkpoints/ml_classifiers \
    --use_stratified_split \
    --classifiers svm rf xgb
```

### Features Extracted from Sequences

The `extract_sequence_features.py` script extracts:
- Basic statistics (mean, std, median, min, max)
- Stage distribution (counts and percentages for each stage 0-5)
- Transition patterns (number of transitions, transition rates)
- Sleep efficiency metrics (sleep/wake percentages)
- REM, deep sleep, light sleep percentages
- Longest sleep bout statistics

Total: ~50 sequence features + 12 PSG features = 62 features

### Classifiers Available

1. **SVM** (`svm`)
   - RBF kernel
   - Class weights: balanced
   - Probability estimates enabled

2. **Random Forest** (`rf`)
   - 200 estimators
   - Max depth: 20
   - Class weights: balanced
   - Min samples split: 10

3. **XGBoost** (`xgb`)
   - 200 estimators
   - Max depth: 6
   - Learning rate: 0.1
   - Scale pos weight: computed from class imbalance

## Model Comparison

After training both neural network and ML classifiers, compare results:

```bash
python code/compare_all_models.py \
    --nn_output_dir checkpoints/comorbidity_fixed \
    --ml_output_dir checkpoints/ml_classifiers \
    --comparison_dir checkpoints/comparison
```

This will:
- Load results from all models
- Create comparison plots
- Generate CSV with metrics comparison
- Print summary table

## Expected Improvements

With the fixes:
1. **Reduced overfitting**: Validation F1 should be closer to training F1
2. **Better generalization**: Model should perform better on test set
3. **More balanced predictions**: Should not predict all positives/negatives
4. **ML baseline**: Traditional ML can serve as baseline for comparison

## Monitoring Training

Watch for:
- **Training vs Validation gap**: Should be smaller (< 0.1 difference)
- **Validation F1**: Should improve steadily, not plateau near 0
- **Per-class performance**: All classes should have non-zero F1
- **Early stopping**: Should trigger based on validation F1 improvement

