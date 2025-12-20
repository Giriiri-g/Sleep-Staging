"""
MESA Dataset Loader for MESA Transformer
========================================

Loads preprocessed MESA tensors and creates sequences of epochs for training.
- Loads .pt files from preprocessed folder (shape: 4 channels x time_samples)
- Extracts only first 3 channels (EEG1, EEG2, EEG3) - excludes 4th channel
- Creates windows of consecutive epochs (seq_len epochs per sample)
- Parses CSV file to get sleep stage labels from 'sleep_stages' column
- Returns sequences compatible with MESATransformer input format

Expected Input:
    Preprocessed .pt files: (4, T) where T is number of time samples
    CSV file with columns: 'mesaid' and 'sleep_stages' (string of epoch labels)

Output:
    Features: (batch, seq_len, num_channels, time_steps) - (batch, 20, 3, 3840)
    Labels: (batch, seq_len) - sleep stage labels per epoch (0-5)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def parse_sleep_stages_from_string(stages_str: str) -> np.ndarray:
    """
    Parse sleep stages string into array of epoch labels.
    
    The sleep_stages column contains a string where each character represents
    an epoch label: 0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM
    
    Args:
        stages_str: String of epoch labels (e.g., "1000112221222111220...")
    
    Returns:
        Array of shape (num_epochs,) with sleep stage labels as integers (0-5)
    """
    # Convert string to array of integers
    labels = np.array([int(c) for c in stages_str if c.isdigit()], dtype=np.int64)
    
    # Filter out any invalid labels (should be 0-5)
    valid_mask = (labels >= 0) & (labels <= 5)
    if not np.all(valid_mask):
        # Replace invalid labels with -1 (unscored)
        labels = np.where(valid_mask, labels, -1)
    
    return labels


class MESADataset(Dataset):
    """
    Dataset for loading preprocessed MESA signals and creating epoch sequences.
    
    Each sample is a sequence of consecutive epochs (seq_len epochs).
    Only first 3 channels are used from the 4-channel preprocessed data.
    """
    
    def __init__(
        self,
        preprocessed_dir: str,
        csv_path: str,
        seq_len: int = 20,
        epoch_seconds: float = 30.0,
        target_fs: float = 128.0,
        stride: Optional[int] = None,
        filter_unscored: bool = True,
    ):
        """
        Initialize MESA Dataset.
        
        Args:
            preprocessed_dir: Directory containing preprocessed .pt files
            csv_path: Path to CSV file with 'mesaid' and 'sleep_stages' columns
            seq_len: Number of consecutive epochs per sample (default: 20)
            epoch_seconds: Length of each epoch in seconds (default: 30s)
            target_fs: Sampling frequency (default: 128 Hz)
            stride: Stride for windowing (default: 1 epoch, i.e., no overlap)
            filter_unscored: Whether to filter out sequences with unscored epochs
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.csv_path = Path(csv_path)
        self.seq_len = seq_len
        self.epoch_seconds = epoch_seconds
        self.target_fs = target_fs
        self.epoch_samples = int(epoch_seconds * target_fs)  # 3840 for 30s @ 128Hz
        self.stride = stride if stride is not None else 1
        self.filter_unscored = filter_unscored
        
        # Stage mapping (6 classes: W, N1, N2, N3, N4, REM)
        self.num_classes = 6
        self.class_names = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
        
        # Load CSV file
        print(f"Loading CSV from {csv_path}...")
        self.csv_df = pd.read_csv(csv_path)
        if 'mesaid' not in self.csv_df.columns or 'sleep_stages' not in self.csv_df.columns:
            raise ValueError("CSV must contain 'mesaid' and 'sleep_stages' columns")
        
        # Create mapping from mesaid to sleep_stages string
        self.mesaid_to_stages = dict(zip(
            self.csv_df['mesaid'].values,
            self.csv_df['sleep_stages'].values
        ))
        
        print(f"Loaded {len(self.mesaid_to_stages)} records from CSV")
        
        # Prepare samples
        self.samples = []
        self._prepare_index()
        
        print(f"Initialized MESA Dataset: {len(self.samples)} samples")
    
    def _get_mesaid_from_filename(self, pt_file: Path) -> Optional[int]:
        """
        Extract mesaid from filename.
        Expected format: mesa-sleep-{mesaid:04d}_preprocessed.pt
        """
        filename = pt_file.stem  # Remove .pt extension
        filename = filename.replace("_preprocessed", "")  # Remove _preprocessed
        
        if "mesa-sleep-" in filename:
            mesaid_str = filename.replace("mesa-sleep-", "")
            try:
                return int(mesaid_str)
            except ValueError:
                return None
        return None
    
    def _prepare_index(self):
        """Prepare index of samples (file_path, start_epoch_idx, labels)"""
        # Find all preprocessed .pt files
        pt_files = list(self.preprocessed_dir.glob("*_preprocessed.pt"))
        
        if not pt_files:
            print(f"Warning: No preprocessed .pt files found in {self.preprocessed_dir}")
            return
        
        print(f"Found {len(pt_files)} preprocessed .pt files")
        
        for pt_file in pt_files:
            # Extract mesaid from filename
            mesaid = self._get_mesaid_from_filename(pt_file)
            
            if mesaid is None:
                print(f"Warning: Could not extract mesaid from {pt_file.name}, skipping")
                continue
            
            # Get sleep stages from CSV
            if mesaid not in self.mesaid_to_stages:
                print(f"Warning: mesaid {mesaid} not found in CSV, skipping {pt_file.name}")
                continue
            
            stages_str = self.mesaid_to_stages[mesaid]
            labels = parse_sleep_stages_from_string(stages_str)
            
            if len(labels) == 0:
                print(f"Warning: No valid labels for {pt_file.name}, skipping")
                continue
            
            # Load tensor to get actual length
            try:
                tensor = torch.load(pt_file)
                # tensor shape: (4, T) where T is number of time samples
                if tensor.dim() != 2 or tensor.shape[0] != 4:
                    print(f"Warning: Unexpected tensor shape {tensor.shape} for {pt_file.name}, expected (4, T), skipping")
                    continue
                
                num_samples = tensor.shape[1]
                num_epochs_available = num_samples // self.epoch_samples
                
                if num_epochs_available < self.seq_len:
                    print(f"Warning: Only {num_epochs_available} epochs available for {pt_file.name} (need {self.seq_len}), skipping")
                    continue
                
                # Adjust labels to match available epochs
                if len(labels) > num_epochs_available:
                    labels = labels[:num_epochs_available]
                elif len(labels) < num_epochs_available:
                    # Pad with -1 (unscored) if labels are shorter
                    padding = np.full(num_epochs_available - len(labels), -1, dtype=np.int64)
                    labels = np.concatenate([labels, padding])
                
            except Exception as e:
                print(f"Warning: Failed to load {pt_file.name}: {e}, skipping")
                continue
            
            # Create sequences with stride
            for start_epoch in range(0, num_epochs_available - self.seq_len + 1, self.stride):
                end_epoch = start_epoch + self.seq_len
                sequence_labels = labels[start_epoch:end_epoch]
                
                # Filter unscored if requested
                if self.filter_unscored and np.any(sequence_labels == -1):
                    continue
                
                self.samples.append((pt_file, start_epoch, sequence_labels))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample: sequence of epochs with labels.
        
        Returns:
            features: (seq_len, num_channels, time_steps) - (20, 3, 3840)
                     Note: DataLoader will batch this to (batch, 20, 3, 3840)
            labels: (seq_len,) - sleep stage labels per epoch (0-5)
                   Note: DataLoader will batch this to (batch, 20)
        """
        pt_file, start_epoch, labels = self.samples[idx]
        
        # Load preprocessed tensor
        tensor = torch.load(pt_file)  # (4, T) where T is number of time samples
        
        # Extract only first 3 channels (EEG1, EEG2, EEG3) - exclude 4th channel (Thor)
        # Ensure we only use first 3 channels even if tensor has more
        if tensor.shape[0] > 3:
            eeg_tensor = tensor[:3, :]  # (3, T) - only first 3 channels
        elif tensor.shape[0] == 3:
            eeg_tensor = tensor  # Already has 3 channels
        else:
            raise ValueError(f"Expected at least 3 channels, got {tensor.shape[0]}")
        
        # Extract sequence of epochs
        start_sample = start_epoch * self.epoch_samples
        end_sample = start_sample + self.seq_len * self.epoch_samples
        
        # Ensure we don't go out of bounds
        if end_sample > eeg_tensor.shape[1]:
            # Pad if necessary
            pad_length = end_sample - eeg_tensor.shape[1]
            padding = torch.zeros(3, pad_length, dtype=eeg_tensor.dtype)
            eeg_tensor = torch.cat([eeg_tensor, padding], dim=1)
        
        # Extract the sequence
        sequence_data = eeg_tensor[:, start_sample:end_sample]  # (3, seq_len * epoch_samples)
        
        # Reshape to (seq_len, num_channels, epoch_samples)
        # First reshape to (3, seq_len, epoch_samples), then permute
        sequence_data = sequence_data.view(3, self.seq_len, self.epoch_samples)
        sequence_data = sequence_data.permute(1, 0, 2)  # (seq_len, 3, epoch_samples)
        
        # Convert labels to tensor
        labels_tensor = torch.LongTensor(labels)
        
        return sequence_data, labels_tensor


def create_mesa_dataloader(
    preprocessed_dir: str,
    csv_path: str,
    seq_len: int = 20,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for MESA dataset.
    
    Args:
        preprocessed_dir: Directory containing preprocessed .pt files
        csv_path: Path to CSV file with 'mesaid' and 'sleep_stages' columns
        seq_len: Number of consecutive epochs per sample
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for MESADataset
    
    Returns:
        DataLoader instance that returns batches:
        - features: (batch, seq_len, num_channels, time_steps) = (batch, 20, 3, 3840)
        - labels: (batch, seq_len) = (batch, 20)
    """
    dataset = MESADataset(
        preprocessed_dir=preprocessed_dir,
        csv_path=csv_path,
        seq_len=seq_len,
        **dataset_kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return dataloader


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("MESA Dataset Loader")
    print("=" * 50)
    
    # Create dataset
    dataset = MESADataset(
        preprocessed_dir=r"C:\mesa",
        csv_path="mesa_final.csv",
        seq_len=20,
        epoch_seconds=30.0,
        target_fs=128.0,
        stride=1,
        filter_unscored=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        features, labels = dataset[0]
        print(f"\nSample features shape: {features.shape}")  # Should be (20, 3, 3840)
        print(f"Sample labels shape: {labels.shape}")  # Should be (20,)
        print(f"Sample labels (first 10): {labels[:10]}")
        print(f"Label values range: {labels.min().item()} to {labels.max().item()}")
    
    # Create DataLoader
    dataloader = create_mesa_dataloader(
        preprocessed_dir=r"C:\mesa",
        csv_path="mesa_final.csv",
        seq_len=20,
        batch_size=4,
        shuffle=True
    )
    
    # Test DataLoader
    print("\nTesting DataLoader:")
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Features shape: {features.shape}")  # (batch, seq_len, 3, time_steps)
        print(f"  Labels shape: {labels.shape}")  # (batch, seq_len)
        print(f"  Labels (first sample, first 10): {labels[0, :10]}")
        if batch_idx >= 2:  # Just test a few batches
            break
