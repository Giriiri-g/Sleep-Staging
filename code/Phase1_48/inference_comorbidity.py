"""
Inference Script for Comorbidity Classifier
============================================
Load a trained model and make predictions on new data.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from comorbidity_classifier_dataset import create_dataloader
from comorbidity_classifier import ComorbidityClassifier


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    # Create model with same architecture
    model = ComorbidityClassifier(
        num_sleep_stages=6,
        embedding_dim=args.get('embedding_dim', 32),
        rnn_hidden_dim=args.get('rnn_hidden_dim', 128),
        rnn_num_layers=args.get('rnn_num_layers', 2),
        rnn_type=args.get('rnn_type', 'LSTM'),
        num_features=args.get('num_features', 12),
        dropout=args.get('dropout', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict(model, dataloader, device, threshold=0.5):
    """Make predictions on a dataset"""
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for sleep_stages, seq_lengths, features, _ in dataloader:
            sleep_stages = sleep_stages.to(device)
            seq_lengths = seq_lengths.to(device)
            features = features.to(device)
            
            # Get predictions
            logits = model(sleep_stages, seq_lengths, features)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    
    return all_probs, all_preds


def main():
    parser = argparse.ArgumentParser(description='Inference with Comorbidity Classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file for inference')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for positive prediction')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='Path to scaler (if different from training)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    
    # Load scaler if provided
    scaler = None
    if args.scaler_path:
        import pickle
        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Create dataloader
    print(f"Loading data from {args.csv_path}...")
    dataloader, dataset = create_dataloader(
        args.csv_path,
        batch_size=args.batch_size,
        shuffle=False,
        scaler=scaler if scaler else dataset.get_scaler() if hasattr(dataset, 'get_scaler') else None
    )
    
    # Make predictions
    print("Making predictions...")
    probs, preds = predict(model, dataloader, device, args.threshold)
    
    # Load original CSV to preserve other columns
    df = pd.read_csv(args.csv_path)
    
    # Add predictions
    target_names = dataset.target_cols
    for i, name in enumerate(target_names):
        df[f'{name}_prob'] = probs[:, i]
        df[f'{name}_pred'] = preds[:, i]
    
    # Save results
    df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    
    # Print summary
    print("\nPrediction Summary:")
    for i, name in enumerate(target_names):
        positive_count = preds[:, i].sum()
        positive_pct = 100 * positive_count / len(preds)
        print(f"  {name}: {positive_count} positive ({positive_pct:.1f}%)")


if __name__ == '__main__':
    main()

