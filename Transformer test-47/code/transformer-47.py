import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================================
# SLEEP TRANSFORMER MODEL (Based on Johns Hopkins Paper)
# ============================================================================

class SleepTransformer(nn.Module):
    """
    Transformer-based model for sleep stage classification
    7 output neurons, 3000 input features (30-second epochs)
    """
    
    def __init__(self, 
                 input_channels=2,  # EEG + EOG
                 sequence_length=3000,  # 30 seconds * 100 Hz
                 n_classes=7,  # User requested 7 output neurons
                 d_model=256,  # embedding dimension
                 n_heads=8,
                 n_layers=4,
                 dropout=0.1):
        
        super(SleepTransformer, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.d_model = d_model
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Calculate sequence length after convolutions
        temp_length = sequence_length
        for stride in [6, 2, 2]:
            temp_length = temp_length // stride
        self.conv_output_length = temp_length
        
        # Linear projection to embedding dimension
        self.input_projection = nn.Linear(256, d_model)
        
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Global average pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _get_positional_encoding(self, seq_len, d_model):
        """Create positional encoding matrix"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        Args: x: (batch_size, channels, sequence_length)
        Returns: (batch_size, n_classes)
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape and project
        x = x.transpose(1, 2)  # (batch, length, channels)
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self._get_positional_encoding(x.size(1), self.d_model).to(x.device)
        x = x + pos_enc
        
        # Transformer
        x = self.transformer(x)
        
        # Global pooling and output
        x = x.transpose(1, 2)  # (batch, d_model, length)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        x = self.output_projection(x)
        
        return x

# ============================================================================
# DATA LOADER FOR SLEEP-EDF
# ============================================================================

class SleepEDFDataset(Dataset):
    def __init__(self, edf_files, target_classes=7):
        self.edf_files = edf_files
        self.target_classes = target_classes
        self.data = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        print("Loading Sleep-EDF data...")
        
        for edf_file in self.edf_files:
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                
                # Get EEG and EOG channels
                eeg_channels = [ch for ch in raw.ch_names if any(x in ch.upper() for x in ['EEG', 'C3', 'C4', 'CZ', 'FPZ'])]
                eog_channels = [ch for ch in raw.ch_names if any(x in ch.upper() for x in ['EOG', 'HORIZONTAL'])]
                
                if not eeg_channels or not eog_channels:
                    continue
                
                # Select channels and resample
                selected_channels = [eeg_channels[0], eog_channels[0]]
                raw.pick_channels(selected_channels)
                if raw.info['sfreq'] != 100:
                    raw.resample(100)
                
                data, _ = raw[:]
                
                # Robust scaling (as per paper)
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx]
                    median_val = np.median(ch_data)
                    q25, q75 = np.percentile(ch_data, [25, 75])
                    iqr = q75 - q25
                    if iqr > 0:
                        data[ch_idx] = (ch_data - median_val) / iqr
                    
                    mad = np.median(np.abs(ch_data - median_val))
                    if mad > 0:
                        data[ch_idx] = np.clip(data[ch_idx], -20*mad, 20*mad)
                
                # Create 30-second epochs
                epoch_length = 3000
                n_epochs = data.shape[1] // epoch_length
                
                for epoch_idx in range(n_epochs):
                    start_idx = epoch_idx * epoch_length
                    end_idx = start_idx + epoch_length
                    epoch_data = data[:, start_idx:end_idx]
                    
                    # Create dummy labels (replace with actual annotations)
                    dummy_label = epoch_idx % self.target_classes
                    
                    self.data.append(epoch_data)
                    self.labels.append(dummy_label)
                
                print(f"Processed {edf_file}: {n_epochs} epochs")
                
            except Exception as e:
                print(f"Error processing {edf_file}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(data_dir, num_epochs=10, batch_size=16, learning_rate=0.000375):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find EDF files
    data_path = Path(data_dir)
    edf_files = list(data_path.glob("*.edf"))
    
    if not edf_files:
        print(f"No EDF files found in {data_dir}")
        return
    
    print(f"Found {len(edf_files)} EDF files")
    
    # Split data
    train_files, val_files = train_test_split(edf_files, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = SleepEDFDataset(train_files)
    val_dataset = SleepEDFDataset(val_files)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SleepTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*correct/total:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'sleep_transformer_model.pth')
    print("Model saved as 'sleep_transformer_model.pth'")
    
    return model

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_sleep_stages(model_path, edf_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SleepTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Movement', 'Unknown']
    
    try:
        # Load and preprocess EDF
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
        
        eeg_channels = [ch for ch in raw.ch_names if any(x in ch.upper() for x in ['EEG', 'C3', 'C4', 'CZ', 'FPZ'])]
        eog_channels = [ch for ch in raw.ch_names if any(x in ch.upper() for x in ['EOG', 'HORIZONTAL'])]
        
        selected_channels = [eeg_channels[0], eog_channels[0]]
        raw.pick_channels(selected_channels)
        
        if raw.info['sfreq'] != 100:
            raw.resample(100)
        
        data, _ = raw[:]
        
        # Robust scaling
        for ch_idx in range(data.shape[0]):
            ch_data = data[ch_idx]
            median_val = np.median(ch_data)
            q25, q75 = np.percentile(ch_data, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                data[ch_idx] = (ch_data - median_val) / iqr
        
        # Predict epochs
        predictions = []
        confidences = []
        epoch_length = 3000
        n_epochs = data.shape[1] // epoch_length
        
        with torch.no_grad():
            for epoch_idx in range(n_epochs):
                start_idx = epoch_idx * epoch_length
                end_idx = start_idx + epoch_length
                
                epoch_data = torch.FloatTensor(data[:, start_idx:end_idx]).unsqueeze(0).to(device)
                output = model(epoch_data)
                
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                predictions.append(class_names[predicted_class])
                confidences.append(confidence)
        
        return predictions, confidences
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

# ============================================================================
# MAIN USAGE
# ============================================================================

if __name__ == "__main__":
    # STEP 1: Train the model (update path to your Sleep-EDF data)
    DATA_DIR = "sleep-edf-database-expanded-1.0.0/sleep-cassette"  # UPDATE THIS PATH
    
    if os.path.exists(DATA_DIR):
        print("Training Sleep Transformer Model...")
        model = train_model(DATA_DIR, num_epochs=5, batch_size=8)  # Reduced for demo
    else:
        print(f"Data directory {DATA_DIR} not found!")
        print("Please update DATA_DIR to point to your Sleep-EDF files")
        
        # Demo: Create and test model with random data
        print("\nDemo: Testing model with random data...")
        model = SleepTransformer()
        sample_input = torch.randn(2, 2, 3000)  # 2 samples, 2 channels, 3000 time points
        output = model(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print("✅ Model architecture working correctly!")
    
    # STEP 2: Inference example (uncomment when you have trained model)
    # predictions, confidences = predict_sleep_stages('sleep_transformer_model.pth', 'your_test_file.edf')
    # if predictions:
    #     print(f"Predicted sleep stages: {predictions[:10]}")  # First 10 epochs
