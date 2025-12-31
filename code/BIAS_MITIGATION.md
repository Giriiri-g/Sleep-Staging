# Bias Mitigation for Class Imbalance

This document describes the bias mitigation techniques implemented to address the highly skewed class distribution in the comorbidity classifier.

## Problem

The target distribution is highly imbalanced:
- **Insomnia**: ~5-8% positive
- **Restless Leg**: ~4-5% positive  
- **Apnea**: ~7-9% positive

This imbalance can lead to:
- Model bias toward predicting the majority class (negative)
- Poor recall for minority classes
- Misleading accuracy metrics

## Implemented Solutions

### 1. **Stratified Sampling** (`--use_stratified_split`)

Ensures balanced class distribution across train/val/test splits by stratifying on the combination of all three binary labels.

**Usage:**
```bash
python train_comorbidity_classifier.py --use_stratified_split ...
```

### 2. **Class-Weighted Loss Functions**

#### a. Weighted BCE Loss (`--loss_type weighted_bce`)

Uses class weights computed from training data to penalize misclassifying minority classes more.

**Weight computation methods:**
- `--class_weight_method balanced`: sklearn-style balancing (n_samples / (2 * n_pos))
- `--class_weight_method inverse`: Inverse frequency (n_neg / n_pos)

**Usage:**
```bash
python train_comorbidity_classifier.py --loss_type weighted_bce --class_weight_method balanced ...
```

#### b. Focal Loss (`--loss_type focal`)

Focal loss addresses class imbalance by:
- Focusing on hard examples
- Down-weighting easy examples
- Using alpha weighting for class balance

**Parameters:**
- `--focal_gamma`: Focusing parameter (default: 2.0). Higher values focus more on hard examples.

**Usage:**
```bash
python train_comorbidity_classifier.py --loss_type focal --focal_gamma 2.0 ...
```

#### c. Standard BCE with Class Weights (`--loss_type bce --class_weight_method balanced`)

Uses PyTorch's built-in `pos_weight` parameter in `BCEWithLogitsLoss`.

**Usage:**
```bash
python train_comorbidity_classifier.py --loss_type bce --class_weight_method balanced ...
```

### 3. **Threshold Optimization** (`--optimize_thresholds`)

Optimizes per-class decision thresholds on the validation set to maximize F1 score or balanced accuracy, rather than using the default 0.5 threshold.

**Parameters:**
- `--threshold_metric`: Metric to optimize (`f1` or `balanced_accuracy`)

**Usage:**
```bash
python train_comorbidity_classifier.py --optimize_thresholds --threshold_metric f1 ...
```

## Recommended Configuration

For highly imbalanced data, use:

```bash
python train_comorbidity_classifier.py \
    --csv_path mesa_final.csv \
    --output_dir checkpoints/comorbidity \
    --use_stratified_split \
    --loss_type focal \
    --focal_gamma 2.0 \
    --optimize_thresholds \
    --threshold_metric f1 \
    --epochs 50 \
    --batch_size 32
```

## Metrics to Monitor

When dealing with class imbalance, focus on:
- **Macro F1**: Average F1 across all classes (better than accuracy)
- **Per-class F1**: Individual class performance
- **ROC AUC**: Area under ROC curve (threshold-independent)
- **Recall**: Ability to find positive cases (critical for rare classes)

## How It Works

1. **Class Weights**: Computed from training data distribution
   - Positive class weight = n_samples / (2 * n_positive) for balanced method
   - Higher weight = more penalty for misclassifying that class

2. **Focal Loss**: 
   - `FL = -alpha * (1 - p_t)^gamma * log(p_t)`
   - `gamma` controls focus on hard examples
   - `alpha` balances class importance

3. **Threshold Optimization**:
   - Evaluates multiple thresholds on validation set
   - Selects threshold that maximizes chosen metric per class
   - Applied during evaluation and inference

## Example Output

With bias mitigation enabled, you should see:
- Improved recall for minority classes
- Better balanced F1 scores across classes
- Optimized thresholds (often < 0.5 for rare classes)

```
Per-class metrics:
  insomnia:
    Precision: 0.XXXX
    Recall: 0.XXXX  # Should be higher with mitigation
    F1: 0.XXXX
    ROC AUC: 0.XXXX
    Threshold: 0.3XX  # Optimized threshold
```

