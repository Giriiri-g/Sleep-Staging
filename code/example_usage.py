"""
Example Usage of Comorbidity Classifier
========================================
Quick example showing how to use the classifier for training and inference.
"""

import torch
from comorbidity_classifier_dataset import create_dataloader
from comorbidity_classifier import ComorbidityClassifier

# Example 1: Load data and create a model
print("Example 1: Loading data and creating model")
print("-" * 50)

# Create dataloader
train_loader, dataset = create_dataloader(
    csv_path='../mesa_final.csv',
    batch_size=8,
    shuffle=True,
    max_seq_len=None  # Use all sequence lengths
)

print(f"Dataset size: {len(dataset)}")
print(f"Feature columns: {dataset.feature_cols}")
print(f"Target columns: {dataset.target_cols}")

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComorbidityClassifier(
    num_sleep_stages=6,
    embedding_dim=32,
    rnn_hidden_dim=128,
    rnn_num_layers=2,
    rnn_type='LSTM',
    num_features=len(dataset.feature_cols),
    dropout=0.3
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Example 2: Forward pass
print("\nExample 2: Forward pass")
print("-" * 50)

# Get a batch
sleep_stages, seq_lengths, features, targets = next(iter(train_loader))
sleep_stages = sleep_stages.to(device)
seq_lengths = seq_lengths.to(device)
features = features.to(device)
targets = targets.to(device)

print(f"Batch shapes:")
print(f"  sleep_stages: {sleep_stages.shape}")
print(f"  seq_lengths: {seq_lengths.shape}")
print(f"  features: {features.shape}")
print(f"  targets: {targets.shape}")

# Forward pass
model.eval()
with torch.no_grad():
    logits = model(sleep_stages, seq_lengths, features)
    probs = model.predict_proba(sleep_stages, seq_lengths, features)
    preds = model.predict(sleep_stages, seq_lengths, features, threshold=0.5)

print(f"\nOutput shapes:")
print(f"  logits: {logits.shape}")
print(f"  probabilities: {probs.shape}")
print(f"  predictions: {preds.shape}")

print(f"\nSample predictions (first sample):")
print(f"  Probabilities: {probs[0].cpu().numpy()}")
print(f"  Predictions: {preds[0].cpu().numpy()}")
print(f"  Ground truth: {targets[0].cpu().numpy()}")

# Example 3: Training setup
print("\nExample 3: Training setup")
print("-" * 50)

import torch.nn as nn
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# One training step
model.train()
optimizer.zero_grad()
logits = model(sleep_stages, seq_lengths, features)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")

print("\n" + "=" * 50)
print("To train the full model, run:")
print("  python train_comorbidity_classifier.py --csv_path ../mesa_final.csv")
print("=" * 50)

