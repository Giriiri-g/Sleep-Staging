"""
Comorbidity Classifier Dataset
==============================
Dataset loader for MESA comorbidity classification task.

Inputs:
- sleep_stages: Sequence of sleep stage labels (string of digits 0-5)
- Other features: ahi_a0h3, ai_all5, odi35, timest1p5, timest2p5, times34p5, 
                  timeremp5, slp_eff5, waso5, plmaslp5, slpprdp5, remlaiip5

Outputs:
- Multiclass binary labels: insomnia, restless leg, apnea (can have multiple)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler


def parse_sleep_stages(stages_str: str) -> np.ndarray:
    """
    Parse sleep stages string into array of epoch labels.
    
    Args:
        stages_str: String of epoch labels (e.g., "1000112221222111220...")
    
    Returns:
        Array of shape (num_epochs,) with sleep stage labels as integers (0-5)
    """
    if pd.isna(stages_str) or stages_str == '':
        return np.array([], dtype=np.int64)
    
    # Convert string to array of integers
    labels = np.array([int(c) for c in str(stages_str) if c.isdigit()], dtype=np.int64)
    
    # Filter out any invalid labels (should be 0-5)
    valid_mask = (labels >= 0) & (labels <= 5)
    if not np.all(valid_mask):
        # Replace invalid labels with 0 (Wake)
        labels = np.where(valid_mask, labels, 0)
    
    return labels


class ComorbidityDataset(Dataset):
    """
    Dataset for comorbidity classification from sleep stages and PSG features.
    """
    
    def __init__(
        self,
        csv_path: str,
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        normalize_features: bool = True,
        scaler: Optional[StandardScaler] = None,
        max_seq_len: Optional[int] = None,
    ):
        """
        Initialize Comorbidity Dataset.
        
        Args:
            csv_path: Path to CSV file
            feature_cols: List of feature column names (excluding sleep_stages)
            target_cols: List of target column names
            normalize_features: Whether to normalize features
            scaler: Pre-fitted scaler (for test set) or None (for train set)
            max_seq_len: Maximum sequence length (None = use all, or pad/truncate)
        """
        self.csv_path = csv_path
        self.normalize_features = normalize_features
        
        # Default feature columns
        if feature_cols is None:
            feature_cols = [
                'ahi_a0h3', 'ai_all5', 'odi35', 'timest1p5', 'timest2p5',
                'times34p5', 'timeremp5', 'slp_eff5', 'waso5', 'plmaslp5',
                'slpprdp5', 'remlaiip5'
            ]
        
        # Default target columns
        if target_cols is None:
            target_cols = ['insomnia', 'restless leg', 'apnea']
        
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        
        # Load CSV
        print(f"Loading CSV from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ['sleep_stages'] + feature_cols + target_cols
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract data
        self.sleep_stages_strs = self.df['sleep_stages'].values
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[target_cols].values.astype(np.float32)
        
        # Parse sleep stages sequences
        print("Parsing sleep stages sequences...")
        self.sleep_stages_sequences = [parse_sleep_stages(s) for s in self.sleep_stages_strs]
        
        # Get sequence lengths
        self.seq_lengths = np.array([len(seq) for seq in self.sleep_stages_sequences])
        
        # Determine max sequence length
        if max_seq_len is None:
            self.max_seq_len = int(self.seq_lengths.max())
        else:
            self.max_seq_len = max_seq_len
        
        print(f"Sequence length stats: min={self.seq_lengths.min()}, "
              f"max={self.seq_lengths.max()}, mean={self.seq_lengths.mean():.1f}")
        print(f"Using max_seq_len={self.max_seq_len}")
        
        # Normalize features
        if normalize_features:
            if scaler is None:
                # Fit scaler on training data
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                # Use provided scaler (for test set)
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Target distribution:")
        for i, col in enumerate(target_cols):
            print(f"  {col}: {self.targets[:, i].sum()} positive ({self.targets[:, i].mean()*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            sleep_stages: (seq_len,) - padded sequence of sleep stage labels
            seq_length: (1,) - actual sequence length (for packing)
            features: (num_features,) - other PSG features
            targets: (num_targets,) - binary labels for comorbidities
        """
        # Get sleep stages sequence
        seq = self.sleep_stages_sequences[idx].copy()
        seq_len = len(seq)
        
        # Pad or truncate sequence
        if seq_len < self.max_seq_len:
            # Pad with 0 (Wake)
            pad_length = self.max_seq_len - seq_len
            seq = np.pad(seq, (0, pad_length), mode='constant', constant_values=0)
        elif seq_len > self.max_seq_len:
            # Truncate
            seq = seq[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Convert to tensors
        sleep_stages = torch.LongTensor(seq)  # (max_seq_len,)
        seq_length = torch.LongTensor([seq_len])  # (1,)
        features = torch.FloatTensor(self.features[idx])  # (num_features,)
        targets = torch.FloatTensor(self.targets[idx])  # (num_targets,)
        
        return sleep_stages, seq_length, features, targets
    
    def get_scaler(self) -> Optional[StandardScaler]:
        """Get the fitted scaler (for use on test set)."""
        return self.scaler
    
    def compute_sample_weights(self, weight_method='inverse_sqrt'):
        """
        Compute sample weights for weighted sampling.
        
        Args:
            weight_method: 'inverse', 'inverse_sqrt', 'balanced', or 'aggressive'
        
        Returns:
            weights: (n_samples,) array of sample weights
        """
        n_samples = len(self.targets)
        weights = np.ones(n_samples, dtype=np.float32)
        
        # For multiclass binary, weight by whether sample has ANY positive label
        has_positive = (self.targets.sum(axis=1) > 0).astype(float)
        n_pos_samples = has_positive.sum()
        n_neg_samples = n_samples - n_pos_samples
        
        if n_pos_samples == 0:
            return weights
        
        if weight_method == 'inverse':
            # Inverse frequency
            pos_weight = n_neg_samples / n_pos_samples
            weights = np.where(has_positive > 0, pos_weight, 1.0)
        elif weight_method == 'inverse_sqrt':
            # Square root of inverse frequency (less aggressive)
            pos_weight = np.sqrt(n_neg_samples / n_pos_samples)
            weights = np.where(has_positive > 0, pos_weight, 1.0)
        elif weight_method == 'balanced':
            # Balanced weights
            pos_weight = n_samples / (2.0 * n_pos_samples)
            weights = np.where(has_positive > 0, pos_weight, 1.0)
        elif weight_method == 'aggressive':
            # Very aggressive: weight positive samples much more
            pos_weight = (n_neg_samples / n_pos_samples) * 5.0  # 5x multiplier
            weights = np.where(has_positive > 0, pos_weight, 1.0)
        elif weight_method == 'per_class':
            # Weight by per-class imbalance (most aggressive)
            max_imbalance = 0
            for i in range(self.targets.shape[1]):
                n_pos = self.targets[:, i].sum()
                if n_pos > 0:
                    imbalance = (n_samples - n_pos) / n_pos
                    max_imbalance = max(max_imbalance, imbalance)
            
            # Weight samples with positive labels
            pos_weight = max_imbalance * 3.0  # 3x multiplier
            weights = np.where(has_positive > 0, pos_weight, 1.0)
        
        return weights


def create_dataloader(
    csv_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    scaler: Optional[StandardScaler] = None,
    max_seq_len: Optional[int] = None,
    use_weighted_sampling: bool = False,
    weight_method: str = 'per_class',
    **dataset_kwargs
) -> Tuple[DataLoader, ComorbidityDataset]:
    """
    Create a DataLoader for the comorbidity dataset.
    
    Args:
        use_weighted_sampling: Use WeightedRandomSampler for imbalanced data
        weight_method: Method for computing sample weights
    
    Returns:
        DataLoader and Dataset instance (to access scaler)
    """
    dataset = ComorbidityDataset(
        csv_path=csv_path,
        scaler=scaler,
        max_seq_len=max_seq_len,
        **dataset_kwargs
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length sequences."""
        sleep_stages_list, seq_lengths_list, features_list, targets_list = zip(*batch)
        
        # Stack sequences (already padded in __getitem__)
        sleep_stages = torch.stack(sleep_stages_list)  # (batch, max_seq_len)
        seq_lengths = torch.cat(seq_lengths_list)  # (batch,)
        features = torch.stack(features_list)  # (batch, num_features)
        targets = torch.stack(targets_list)  # (batch, num_targets)
        
        return sleep_stages, seq_lengths, features, targets
    
    # Create weighted sampler if requested
    sampler = None
    if use_weighted_sampling and shuffle:
        sample_weights = dataset.compute_sample_weights(weight_method=weight_method)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow replacement to ensure positive samples
        )
        shuffle = False  # Don't shuffle when using sampler
        print(f"Using weighted sampling (method: {weight_method})")
        print(f"  Positive sample weight: {sample_weights[sample_weights > 1.0].mean():.2f}")
        print(f"  Negative sample weight: {sample_weights[sample_weights <= 1.0].mean():.2f}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, dataset

