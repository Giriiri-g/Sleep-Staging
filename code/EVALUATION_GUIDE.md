# Evaluation Guide for Comorbidity Classifier

## Quick Start

### Evaluate Existing Checkpoint

If you've already trained a model and want to evaluate it without retraining:

```bash
python code/evaluate_checkpoint.py \
    --output_dir checkpoints/comorbidity \
    --optimize_thresholds \
    --eval_splits test val
```

This will:
- Load the latest checkpoint (or best_model.pt if available)
- Optimize thresholds on validation set
- Evaluate on test and validation sets
- Generate all visualizations

### Continue Training

If training was interrupted, you can resume from a checkpoint:

```bash
python code/train_comorbidity_classifier.py \
    --csv_path mesa_final.csv \
    --output_dir checkpoints/comorbidity \
    --resume checkpoints/comorbidity/checkpoint_epoch_50.pt \
    --use_stratified_split \
    --loss_type focal \
    --focal_gamma 2.0 \
    --optimize_thresholds
```

## Generated Visualizations

After evaluation, you'll find these files in the output directory:

### For Each Split (train/val/test):
- `eval_{split}/confusion_matrices.png` - Confusion matrices for each class
- `eval_{split}/roc_curves.png` - ROC curves showing AUC for each class
- `eval_{split}/pr_curves.png` - Precision-Recall curves
- `eval_{split}/class_performance.png` - Bar chart comparing metrics across classes
- `eval_{split}/results.json` - Detailed metrics in JSON format
- `eval_{split}/predictions.npz` - Raw predictions and probabilities

### Overall:
- `training_history.png` - Training curves (loss, accuracy, F1, ROC AUC)
- `test_results.json` - Summary of test set performance
- `training_history.json` - Full training history data

## Understanding the Results

### Key Metrics:
- **Macro F1**: Average F1 score across all classes (better than accuracy for imbalanced data)
- **ROC AUC**: Area under ROC curve (threshold-independent, good for imbalanced data)
- **Per-class F1**: Individual class performance
- **Optimal Thresholds**: Per-class thresholds optimized on validation set (often < 0.5 for rare classes)

### Common Issues:

1. **All predictions are negative (F1 = 0)**
   - Model is too conservative
   - Try: Lower learning rate, different loss function, or threshold optimization
   - Check if ROC AUC > 0.5 (model has some signal)

2. **Low recall for minority classes**
   - Expected with severe imbalance
   - Use threshold optimization to improve recall
   - Consider using focal loss with higher gamma

3. **High accuracy but low F1**
   - Model is predicting mostly negatives (majority class)
   - Focus on macro F1 and ROC AUC instead of accuracy

## Troubleshooting

### Checkpoint Not Found Error
If you get `FileNotFoundError` for `best_model.pt`:
- The script will automatically use the latest checkpoint
- Or manually specify: `--checkpoint checkpoints/comorbidity/checkpoint_epoch_50.pt`

### Missing Visualizations
- Ensure matplotlib and seaborn are installed: `pip install matplotlib seaborn`
- Check that evaluation completed successfully (no errors)

### Low Performance
- Try different loss functions (`--loss_type focal` or `--loss_type weighted_bce`)
- Increase training epochs
- Adjust learning rate
- Use threshold optimization (`--optimize_thresholds`)

