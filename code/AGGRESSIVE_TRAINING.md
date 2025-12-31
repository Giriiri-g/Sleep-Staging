# Aggressive Training for Severe Class Imbalance

## Problem
The model is predicting all negatives due to severe class imbalance (~5% positive rate). This guide provides aggressive measures to force the model to learn.

## Recommended Training Command

```bash
python code/train_comorbidity_classifier.py \
    --csv_path mesa_final.csv \
    --output_dir checkpoints/comorbidity_aggressive \
    --use_stratified_split \
    --use_weighted_sampling \
    --weight_method per_class \
    --loss_type focal \
    --focal_gamma 3.0 \
    --class_weight_method aggressive \
    --class_weight_multiplier 2.0 \
    --optimize_thresholds \
    --early_stop_metric macro_f1 \
    --early_stop_patience 15 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

## Key Features

### 1. **Weighted Sampling** (`--use_weighted_sampling`)
- Uses `WeightedRandomSampler` to oversample positive examples
- Ensures each batch has positive examples
- Methods:
  - `per_class`: Most aggressive, weights by worst class imbalance
  - `aggressive`: 5x multiplier on positive samples
  - `balanced`: Moderate weighting

### 2. **Aggressive Class Weights** (`--class_weight_method aggressive`)
- Options:
  - `balanced`: Standard sklearn-style (n_samples / (2 * n_pos))
  - `inverse`: Inverse frequency (n_neg / n_pos)
  - `aggressive`: 3x inverse frequency
  - `very_aggressive`: 5x inverse frequency
- Combined with `--class_weight_multiplier` for even more weight

### 3. **Focal Loss** (`--loss_type focal --focal_gamma 3.0`)
- Higher gamma (3.0) focuses more on hard examples
- Alpha weighting balances classes
- Better than BCE for severe imbalance

### 4. **Early Stopping on F1** (`--early_stop_metric macro_f1`)
- Stops based on F1 score instead of loss
- Prevents overfitting to majority class
- `--early_stop_patience 15`: Wait 15 epochs for improvement

### 5. **Threshold Optimization** (`--optimize_thresholds`)
- Finds optimal per-class thresholds
- Often finds thresholds < 0.5 for rare classes
- Maximizes F1 or balanced accuracy

## Monitoring Training

Watch for these signs:
- **Positive predictions**: Should see some positive predictions early (epoch 1-5)
- **ROC AUC**: Should be > 0.5 (even if F1 is 0)
- **Recall**: Should increase over time
- **Per-class F1**: Should improve from 0

## If Model Still Predicts All Negatives

Try even more aggressive settings:

```bash
python code/train_comorbidity_classifier.py \
    --csv_path mesa_final.csv \
    --output_dir checkpoints/comorbidity_very_aggressive \
    --use_stratified_split \
    --use_weighted_sampling \
    --weight_method per_class \
    --loss_type weighted_bce \
    --class_weight_method very_aggressive \
    --class_weight_multiplier 5.0 \
    --optimize_thresholds \
    --early_stop_metric macro_recall \
    --early_stop_patience 20 \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4
```

## Troubleshooting

1. **Still all negatives after 10 epochs**
   - Increase `--class_weight_multiplier` to 3.0 or 5.0
   - Use `--weight_method per_class`
   - Try `--loss_type weighted_bce` with `very_aggressive`

2. **Model overfitting to positives**
   - Reduce `--class_weight_multiplier`
   - Use `--weight_method balanced` instead of `per_class`
   - Increase dropout

3. **Low precision but high recall**
   - Use threshold optimization
   - Increase threshold for better precision
   - This is expected with severe imbalance

4. **Training unstable**
   - Reduce learning rate: `--lr 5e-4`
   - Smaller batch size: `--batch_size 16`
   - Lower `--class_weight_multiplier`

## Expected Results

With aggressive training, you should see:
- **Epoch 1-5**: Some positive predictions (F1 > 0)
- **Epoch 10-20**: Improving recall and F1
- **Final**: 
  - Macro F1: 0.1-0.3 (good for 5% positive rate)
  - Per-class recall: 0.3-0.6
  - ROC AUC: 0.6-0.8

Remember: With 5% positive rate, perfect recall would mean 95% false positives. Focus on F1 and ROC AUC, not accuracy.

