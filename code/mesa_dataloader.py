"""
MESA Dataset Loader for MESA Transformer
========================================

Loads preprocessed MESA tensors and creates sequences of epochs for training.
- Loads .pt files from preprocessed folder
- Extracts only first 3 channels (EEG1, EEG2, EEG3) - excludes Thor
- Creates windows of consecutive epochs (seq_len epochs per sample)
- Parses XML annotations to get sleep stage labels
- Returns sequences compatible with MESATransformer input format

Expected Input:
    Preprocessed .pt files: (4, T) where channels are [EEG1, EEG2, EEG3, Thor]
    XML annotation files: Sleep stage annotations

Output:
    Features: (seq_len, 3, time_steps) - sequences of epochs with 3 channels
    Labels: (seq_len,) - sleep stage labels per epoch
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET


def find_annotation_file(preprocessed_path: Path, annotation_dir: Path) -> Optional[Path]:
    """
    Find corresponding XML annotation file for a preprocessed .pt file.
    
    Naming convention:
    - Preprocessed: {mesa-sleep-{nsrrid}}_preprocessed.pt
    - XML: {nsrrid}-nsrr.xml
    """
    base_name = preprocessed_path.stem.replace("_preprocessed", "")
    
    # Primary pattern: mesa-sleep-{nsrrid}
    if "mesa-sleep-" in base_name:
        nsrrid = base_name.replace("mesa-sleep-", "")
        xml_filename = f"{nsrrid}-nsrr.xml"
        xml_path = annotation_dir / xml_filename
        if xml_path.exists():
            return xml_path
    
    # Fallback: any XML whose name contains the base name prefix
    for f in annotation_dir.glob("*.xml"):
        if base_name[:17] in f.name or nsrrid in f.name:
            return f
    
    return None


def parse_sleep_stages_from_xml(xml_path: Path, epoch_seconds: float = 30.0) -> Optional[np.ndarray]:
    """
    Parse XML hypnogram and return array of sleep stage labels per epoch.
    
    Supports both MESA XML formats:
    1. NSRR format: "Wake|0", "Stage 1 sleep|1", etc.
    2. MESA format: "0", "1", "2", etc. in EventConcept
    
    Args:
        xml_path: Path to XML annotation file
        epoch_seconds: Length of each epoch in seconds (default: 30s)
    
    Returns:
        Array of shape (num_epochs,) with sleep stage labels:
        0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM, -1=Unscored
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        
        # Stage mapping for MESA (6 classes: W, N1, N2, N3, N4, REM)
        # Support both NSRR format and simple numeric format
        stage_mapping_nsrr = {
            "Wake|0": 0,
            "Stage 1 sleep|1": 1,
            "Stage 2 sleep|2": 2,
            "Stage 3 sleep|3": 3,
            "Stage 4 sleep|4": 4,
            "REM sleep|5": 5,
        }
        
        stage_mapping_simple = {
            "0": 0,  # Wake
            "1": 1,  # N1
            "2": 2,  # N2
            "3": 3,  # N3
            "4": 4,  # N4
            "5": 5,  # REM
        }
        
        # Collect all sleep stage events
        events = []
        for event in root.findall(".//ScoredEvent"):
            start_el = event.find("Start")
            duration_el = event.find("Duration")
            type_el = event.find("EventType")
            concept_el = event.find("EventConcept")
            
            if (
                start_el is None
                or duration_el is None
                or type_el is None
                or concept_el is None
            ):
                continue
            
            event_type = type_el.text or ""
            if not event_type.startswith("Stages|"):
                continue
            
            start = float(start_el.text)
            duration = float(duration_el.text)
            stage_str = concept_el.text or ""
            
            # Try NSRR format first, then simple numeric format
            label = stage_mapping_nsrr.get(stage_str, -1)
            if label == -1:
                # Try extracting numeric code from format like "Stage 1 sleep|1"
                if "|" in stage_str:
                    stage_code = stage_str.split("|")[-1].strip()
                    label = stage_mapping_simple.get(stage_code, -1)
                else:
                    # Direct numeric format
                    label = stage_mapping_simple.get(stage_str.strip(), -1)
            
            if label != -1:
                events.append((start, duration, label))
        
        if not events:
            return None
        
        # Find total duration and create epoch labels
        max_time = max(start + duration for start, duration, _ in events)
        num_epochs = int(np.ceil(max_time / epoch_seconds))
        
        # Initialize labels array with -1 (unscored)
        labels = np.full(num_epochs, -1, dtype=np.int64)
        
        # Fill in labels for each epoch
        for start, duration, label in events:
            start_epoch = int(start / epoch_seconds)
            num_epochs_in_event = int(np.ceil(duration / epoch_seconds))
            
            for i in range(num_epochs_in_event):
                epoch_idx = start_epoch + i
                if epoch_idx < num_epochs:
                    labels[epoch_idx] = label
        
        return labels
    
    except Exception as e:
        print(f"Warning: Failed to parse XML {xml_path}: {e}")
        return None


class MESADataset(Dataset):
    """
    Dataset for loading preprocessed MESA signals and creating epoch sequences.
    
    Each sample is a sequence of consecutive epochs (seq_len epochs).
    Only first 3 channels (EEG1, EEG2, EEG3) are used - Thor is excluded.
    """
    
    def __init__(
        self,
        preprocessed_dir: str,
        annotation_dir: str,
        seq_len: int = 20,
        epoch_seconds: float = 30.0,
        target_fs: float = 128.0,
        stride: Optional[int] = None,
        filter_unscored: bool = True,
        skip_initial_wake: bool = False,
    ):
        """
        Initialize MESA Dataset.
        
        Args:
            preprocessed_dir: Directory containing preprocessed .pt files
            annotation_dir: Directory containing XML annotation files
            seq_len: Number of consecutive epochs per sample (default: 20)
            epoch_seconds: Length of each epoch in seconds (default: 30s)
            target_fs: Sampling frequency (default: 128 Hz)
            stride: Stride for windowing (default: 1 epoch, i.e., no overlap)
            filter_unscored: Whether to filter out sequences with unscored epochs
            skip_initial_wake: Whether to skip initial wake epochs (not implemented yet)
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.annotation_dir = Path(annotation_dir)
        self.seq_len = seq_len
        self.epoch_seconds = epoch_seconds
        self.target_fs = target_fs
        self.epoch_samples = int(epoch_seconds * target_fs)  # 3840 for 30s @ 128Hz
        self.stride = stride if stride is not None else 1
        self.filter_unscored = filter_unscored
        self.skip_initial_wake = skip_initial_wake
        
        # Stage mapping (6 classes: W, N1, N2, N3, N4, REM)
        self.num_classes = 6
        self.class_names = ['W', 'N1', 'N2', 'N3', 'N4', 'REM']
        
        # Prepare samples
        self.samples = []
        self._prepare_index()
        
        print(f"Initialized MESA Dataset: {len(self.samples)} samples")
    
    def _prepare_index(self):
        """Prepare index of samples (file_path, start_epoch_idx, labels)"""
        # Find all preprocessed .pt files
        pt_files = list(self.preprocessed_dir.glob("*_preprocessed.pt"))
        
        if not pt_files:
            print(f"Warning: No preprocessed .pt files found in {self.preprocessed_dir}")
            return
        
        for pt_file in pt_files:
            # Find corresponding XML annotation
            xml_path = find_annotation_file(pt_file, self.annotation_dir)
            
            if xml_path is None:
                print(f"Warning: No XML found for {pt_file.name}, skipping")
                continue
            
            # Parse sleep stage labels
            labels = parse_sleep_stages_from_xml(xml_path, self.epoch_seconds)
            
            if labels is None:
                print(f"Warning: Failed to parse labels for {pt_file.name}, skipping")
                continue
            
            # Load tensor to get actual length
            try:
                tensor = torch.load(pt_file)
                # tensor shape: (4, T) where channels are [EEG1, EEG2, EEG3, Thor]
                num_samples = tensor.shape[1]
                num_epochs_available = num_samples // self.epoch_samples
                
                # Adjust labels to match available epochs
                if len(labels) > num_epochs_available:
                    labels = labels[:num_epochs_available]
                elif len(labels) < num_epochs_available:
                    # Pad with -1 (unscored)
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
            features: (seq_len, 3, time_steps) - sequence of epochs with 3 channels (EEG1, EEG2, EEG3)
            labels: (seq_len,) - sleep stage labels per epoch
        """
        pt_file, start_epoch, labels = self.samples[idx]
        
        # Load preprocessed tensor
        tensor = torch.load(pt_file)  # (4, T) where channels are [EEG1, EEG2, EEG3, Thor]
        
        # Extract only first 3 channels (EEG1, EEG2, EEG3)
        eeg_tensor = tensor[:3, :]  # (3, T)
        
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
        
        # Reshape to (seq_len, 3, epoch_samples)
        sequence_data = sequence_data.view(3, self.seq_len, self.epoch_samples)
        sequence_data = sequence_data.permute(1, 0, 2)  # (seq_len, 3, epoch_samples)
        
        # Convert labels to tensor
        labels_tensor = torch.LongTensor(labels)
        
        return sequence_data, labels_tensor


def create_mesa_dataloader(
    preprocessed_dir: str,
    annotation_dir: str,
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
        annotation_dir: Directory containing XML annotation files
        seq_len: Number of consecutive epochs per sample
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for MESADataset
    
    Returns:
        DataLoader instance
    """
    dataset = MESADataset(
        preprocessed_dir=preprocessed_dir,
        annotation_dir=annotation_dir,
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
        preprocessed_dir="MESA_preprocessed",
        annotation_dir="path/to/annotations",
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
        print(f"Features shape: {features.shape}")  # Should be (20, 3, 3840)
        print(f"Labels shape: {labels.shape}")  # Should be (20,)
        print(f"Sample labels: {labels[:5]}")
    
    # Create DataLoader
    dataloader = create_mesa_dataloader(
        preprocessed_dir="MESA_preprocessed",
        annotation_dir="path/to/annotations",
        seq_len=20,
        batch_size=4,
        shuffle=True
    )
    
    # Test DataLoader
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Features shape: {features.shape}")  # (batch, seq_len, 3, time_steps)
        print(f"  Labels shape: {labels.shape}")  # (batch, seq_len)
        if batch_idx >= 2:  # Just test a few batches
            break

