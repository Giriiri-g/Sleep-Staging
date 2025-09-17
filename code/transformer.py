import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EEGDataset(Dataset):
    """Custom dataset for EEG signal data"""
    def __init__(self, data_dir, sequence_length=256, overlap=0.5):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.data, self.labels = self._load_data()
        self.scaler = StandardScaler()
        self.data = self._normalize_data()
        
    def _load_data(self):
        """Load EEG epochs and labels from the dataset"""
        all_data = []
        all_labels = []
        
        # Find all epoch files
        epoch_files = list(self.data_dir.glob("**/processed/*epochs.npy"))
        label_files = list(self.data_dir.glob("**/processed/*labels.npy"))
        
        print(f"Found {len(epoch_files)} epoch files and {len(label_files)} label files")
        
        for epoch_file in epoch_files:
            # Find corresponding label file
            subject_id = epoch_file.stem.split('-')[0]  # Extract subject ID
            label_file = None
            for lf in label_files:
                if lf.stem.split('-')[0] == subject_id:
                    label_file = lf
                    break
            
            if label_file and epoch_file.exists() and label_file.exists():
                try:
                    epochs = np.load(epoch_file)
                    labels = np.load(label_file)
                    
                    print(f"Loaded {subject_id}: epochs shape {epochs.shape}, labels shape {labels.shape}")
                    
                    # Ensure labels match epochs
                    min_len = min(len(epochs), len(labels))
                    epochs = epochs[:min_len]
                    labels = labels[:min_len]
                    
                    all_data.append(epochs)
                    all_labels.append(labels)
                    
                except Exception as e:
                    print(f"Error loading {epoch_file}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No valid data files found. Check your data directory structure.")
        
        # Concatenate all data
        data = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total dataset: {data.shape[0]} samples, {data.shape[1]} channels, {data.shape[2]} time points")
        
        return data, labels
    
    def _normalize_data(self):
        """Normalize the data"""
        # Reshape for normalization (samples * channels, time_points)
        original_shape = self.data.shape
        reshaped_data = self.data.reshape(-1, original_shape[-1])
        
        # Fit and transform
        normalized_data = self.scaler.fit_transform(reshaped_data)
        
        # Reshape back to original
        return normalized_data.reshape(original_shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the epoch data and reshape if needed
        epoch = self.data[idx]
        label = self.labels[idx]
        
        # If epoch has multiple channels, flatten or select channels
        if len(epoch.shape) == 2:  # (channels, time_points)
            # For transformer, we'll treat each channel as a feature
            epoch = epoch.T  # (time_points, channels)
        
        return torch.FloatTensor(epoch), torch.LongTensor([label]).squeeze()

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LightweightTransformer(nn.Module):
    """Lightweight Transformer for EEG anomaly detection"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, 
                 num_classes=2, max_seq_length=1000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Lightweight transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Smaller feedforward dimension
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Transformer encoding
        if return_attention:
            # For attention visualization
            encoded = x
            for layer in self.transformer_encoder.layers:
                encoded, attention_weights = layer(encoded, need_weights=True)
        else:
            encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attention_weights
        return output

class TimeShapExplainer:
    """Time-based SHAP explainer for transformer model"""
    def __init__(self, model, background_data):
        self.model = model
        self.model.eval()
        self.background_data = background_data
        
    def explain_instance(self, instance, target_class=None):
        """Explain a single instance using time-based approach"""
        self.model.eval()
        
        def model_wrapper(x):
            """Wrapper for SHAP that handles tensor conversion"""
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                output = self.model(x)
                return torch.softmax(output, dim=-1).cpu().numpy()
        
        # Create explainer
        explainer = shap.DeepExplainer(model_wrapper, self.background_data)
        
        # Get SHAP values
        shap_values = explainer.shap_values(instance.unsqueeze(0))
        
        return shap_values
    
    def plot_time_explanation(self, instance, shap_values, channel_names=None, 
                            target_class=1, figsize=(15, 8)):
        """Plot time-based explanation"""
        if isinstance(shap_values, list):
            shap_vals = shap_values[target_class][0]  # Get values for target class
        else:
            shap_vals = shap_values[0]
        
        instance_np = instance.cpu().numpy() if torch.is_tensor(instance) else instance
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot original signal
        if len(instance_np.shape) == 2:
            for i in range(min(3, instance_np.shape[1])):  # Plot first 3 channels
                channel_name = f'Channel {i+1}' if channel_names is None else channel_names[i]
                axes[0].plot(instance_np[:, i], label=channel_name, alpha=0.7)
        else:
            axes[0].plot(instance_np, label='Signal', alpha=0.7)
        
        axes[0].set_title('Original EEG Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot SHAP values
        if len(shap_vals.shape) == 2:
            # Average across channels or plot each channel
            shap_mean = np.mean(shap_vals, axis=1)
            axes[1].plot(shap_mean, color='red', linewidth=2)
            
            # Also show individual channels with lower alpha
            for i in range(min(3, shap_vals.shape[1])):
                axes[1].plot(shap_vals[:, i], alpha=0.3)
        else:
            axes[1].plot(shap_vals, color='red', linewidth=2)
        
        axes[1].set_title(f'SHAP Values (Class {target_class} - Anomaly Detection)')
        axes[1].set_xlabel('Time Points')
        axes[1].set_ylabel('SHAP Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
    """Train the transformer model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_eeg_transformer.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, class_names=['Normal', 'Anomaly']):
    """Evaluate the trained model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return all_preds, all_targets

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    # Configuration
    DATA_DIR = "sleep-edf-database-expanded-1.0.0/sleep-cassette"  # Update this path
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 256
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    print("Loading EEG dataset...")
    try:
        dataset = EEGDataset(DATA_DIR, sequence_length=SEQUENCE_LENGTH)
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        
        # Create anomaly labels based on sleep stages
        # Assuming labels: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        # We'll treat Wake and N1 as potential anomalies (light sleep/wake)
        anomaly_labels = (dataset.labels <= 1).astype(int)  # 0=normal, 1=anomaly
        
        # Update dataset labels for binary classification
        dataset.labels = anomaly_labels
        
        print(f"Anomaly distribution: {np.bincount(anomaly_labels)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset for demonstration...")
        
        # Create synthetic EEG-like data for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_channels = 2
        seq_length = 256
        
        # Generate synthetic data
        synthetic_data = np.random.randn(n_samples, seq_length, n_channels)
        
        # Add some patterns for anomalies
        anomaly_indices = np.random.choice(n_samples, size=n_samples//4, replace=False)
        for idx in anomaly_indices:
            # Add high frequency noise for anomalies
            synthetic_data[idx] += 2 * np.random.randn(seq_length, n_channels)
        
        synthetic_labels = np.zeros(n_samples)
        synthetic_labels[anomaly_indices] = 1
        
        # Create a mock dataset
        class MockDataset:
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]]).squeeze()
        
        dataset = MockDataset(synthetic_data, synthetic_labels)
        print(f"Synthetic dataset created: {len(dataset)} samples")
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42, 
                                         stratify=dataset.labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                       stratify=dataset.labels[temp_idx])
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Get input dimensions
    sample_data, _ = dataset[0]
    input_dim = sample_data.shape[1] if len(sample_data.shape) == 2 else 1
    
    print(f"Input dimensions: {sample_data.shape}, Feature dim: {input_dim}")
    
    # Initialize model
    model = LightweightTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_seq_length=SEQUENCE_LENGTH,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_eeg_transformer.pth'))
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, targets = evaluate_model(model, test_loader)
    
    # XAI Analysis
    print("\nGenerating explanations...")
    
    # Get background data for SHAP
    background_samples = []
    for i, (data, _) in enumerate(train_loader):
        background_samples.append(data)
        if i >= 5:  # Use first few batches as background
            break
    background_data = torch.cat(background_samples, dim=0)[:50].to(device)  # 50 samples
    
    # Initialize explainer
    explainer = TimeShapExplainer(model, background_data)
    
    # Explain a few test instances
    test_samples = []
    test_labels = []
    for data, labels in test_loader:
        test_samples.append(data)
        test_labels.append(labels)
        break
    
    test_data = test_samples[0].to(device)
    test_targets = test_labels[0]
    
    # Explain anomalous instances
    anomaly_indices = torch.where(test_targets == 1)[0]
    if len(anomaly_indices) > 0:
        for i, idx in enumerate(anomaly_indices[:3]):  # Explain first 3 anomalies
            print(f"\nExplaining anomaly {i+1}...")
            instance = test_data[idx]
            
            try:
                shap_values = explainer.explain_instance(instance, target_class=1)
                fig = explainer.plot_time_explanation(
                    instance, shap_values, target_class=1,
                    figsize=(15, 8)
                )
                plt.suptitle(f'Anomaly Explanation {i+1}', fontsize=14)
                plt.show()
            except Exception as e:
                print(f"Error generating explanation: {e}")
                continue
    
    print("\nAnalysis complete!")
    print(f"Model saved as 'best_eeg_transformer.pth'")

if __name__ == "__main__":
    main()