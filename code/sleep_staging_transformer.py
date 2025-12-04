"""
Sleep Staging Transformer - Full Implementation
===============================================

A hierarchical transformer model for sleep stage classification using PSG data from EDF files.
Includes data loading, feature extraction, training, validation, and checkpointing.

Author: Sleep Staging Team
"""

import os
import re
import sys
import warnings
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy import signal
from scipy.signal import spectrogram

# Suppress all warnings
os.environ['MNE_LOGGING_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')
import mne
mne.set_log_level('ERROR')

# Color codes for terminal output
class Colors:
    """ANSI color codes for colored terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'


def print_colored(message: str, color: str = Colors.ENDC, bold: bool = False):
    """Print colored message to terminal"""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{message}{Colors.ENDC}")


def print_debug(message: str, level: str = "INFO"):
    """Print debug message with color coding based on level"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if level == "INFO":
        print_colored(f"[{timestamp}] [INFO] {message}", Colors.OKCYAN)
    elif level == "SUCCESS":
        print_colored(f"[{timestamp}] [SUCCESS] {message}", Colors.OKGREEN, bold=True)
    elif level == "WARNING":
        print_colored(f"[{timestamp}] [WARNING] {message}", Colors.WARNING)
    elif level == "ERROR":
        print_colored(f"[{timestamp}] [ERROR] {message}", Colors.FAIL, bold=True)
    elif level == "TRAIN":
        print_colored(f"[{timestamp}] [TRAIN] {message}", Colors.OKBLUE)
    elif level == "VAL":
        print_colored(f"[{timestamp}] [VAL] {message}", Colors.MAGENTA)
    else:
        print(f"[{timestamp}] [{level}] {message}")


# ============================================================================
# Model Architecture
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding module compatible with Transformer encoders.
    Implements sinusoidal positional encoding as described in "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LocalTransformerEncoder(nn.Module):
    """Local-Level Transformer Encoder (Bottom Tier)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048
    ):
        super(LocalTransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Mean pooling
        return x


class GlobalTransformerEncoder(nn.Module):
    """Global-Level Transformer Encoder (Middle Tier)"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048
    ):
        super(GlobalTransformerEncoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x


class HierarchicalTransformerModel(nn.Module):
    """
    Hierarchical Transformer Model for Sleep Staging.
    
    Processes sequential spectrogram-like inputs hierarchically:
    1. Local-Level: Each segment is processed independently
    2. Global-Level: Sequence of segment embeddings is processed
    3. Prediction Head: Each global embedding is classified
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers_local: int,
        num_encoder_layers_global: int,
        num_classes: int,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        fc_hidden_dim: int = 512
    ):
        super(HierarchicalTransformerModel, self).__init__()
        
        self.local_encoder = LocalTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers_local,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        
        self.global_encoder = GlobalTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers_global,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, time_steps, input_dim = x.shape
        
        # Reshape for local encoder
        x = x.view(batch_size * num_segments, time_steps, input_dim)
        
        # Process each segment through local encoder
        local_embeddings = self.local_encoder(x)
        
        # Reshape back to sequence
        local_embeddings = local_embeddings.view(batch_size, num_segments, -1)
        
        # Process through global encoder
        global_embeddings = self.global_encoder(local_embeddings)
        
        # Apply prediction head
        predictions = self.prediction_head(global_embeddings)
        
        return predictions


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

class SleepEDFDataset(Dataset):
    """
    Dataset for loading and processing Sleep-EDF database files.
    Each sample consists of a sequence of 30-second epochs (segments).
    """
    
    def __init__(
        self,
        folder_path: str,
        segment_size: int = 10,  # Number of consecutive epochs per sample
        epoch_seconds: int = 30,
        target_fs: int = 100,
        use_spectrogram: bool = True,
        nfft: int = 256,
        overlap: int = 128,
        channels: Optional[List[str]] = None,
        filter_unscored: bool = True
    ):
        self.folder_path = folder_path
        self.segment_size = segment_size
        self.epoch_seconds = epoch_seconds
        self.target_fs = target_fs
        self.use_spectrogram = use_spectrogram
        self.nfft = nfft
        self.overlap = overlap
        self.filter_unscored = filter_unscored
        
        # Sleep stage mapping
        self.stage_mapping = {
            'Sleep stage W': 0,
            'Sleep stage R': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3': 4,
            'Sleep stage 4': 5,
            'Sleep stage ?': 6  # Unscored
        }
        self.num_classes = 6 if filter_unscored else 7
        
        # Default channels (7 PSG channels)
        self.channels = channels or [
            'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 
            'EMG submental', 'Temp rectal', 'Resp oro-nasal', 'Event marker'
        ]
        
        # Prepare data index
        self.samples = []
        self.sfreq_cache = {}
        self._prepare_index()
        
        print_debug(f"Dataset initialized with {len(self.samples)} samples", "INFO")
        print_debug(f"Number of classes: {self.num_classes}", "INFO")
    
    def _find_annotation_file(self, psg_filename: str) -> Optional[str]:
        """Find corresponding hypnogram file for a PSG file"""
        base = psg_filename[:6]
        pattern = re.compile(rf'{base}..-Hypnogram.edf')
        for f in os.listdir(self.folder_path):
            if pattern.fullmatch(f):
                return os.path.join(self.folder_path, f)
        return None
    
    def _prepare_index(self):
        """Prepare index of all valid samples (sequences of consecutive epochs)"""
        print_debug("Preparing dataset index...", "INFO")
        
        # First pass: collect all epochs with their file, start sample, and label
        all_epochs = []
        
        for psg_file in os.listdir(self.folder_path):
            if not psg_file.endswith('-PSG.edf'):
                continue
            
            psg_path = os.path.join(self.folder_path, psg_file)
            hyp_path = self._find_annotation_file(psg_file)
            
            if not hyp_path:
                print_debug(f"No annotation file for {psg_file}, skipping", "WARNING")
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
                    sfreq = int(raw.info['sfreq'])
                    self.sfreq_cache[psg_file] = sfreq
                    
                    annotations = mne.read_annotations(hyp_path)
                    
                    epoch_samples = int(self.epoch_seconds * sfreq)
                    
                    for desc, onset, duration in zip(
                        annotations.description, 
                        annotations.onset, 
                        annotations.duration
                    ):
                        label = self.stage_mapping.get(desc, 6)
                        
                        # Filter unscored if requested
                        if self.filter_unscored and label == 6:
                            continue
                        
                        full_epochs = int(duration // self.epoch_seconds)
                        start_sample = int(onset * sfreq)
                        
                        for i in range(full_epochs):
                            epoch_start = start_sample + i * epoch_samples
                            epoch_end = epoch_start + epoch_samples
                            
                            if epoch_end <= raw.n_times:
                                all_epochs.append((psg_file, epoch_start, label))
                    
            except Exception as e:
                print_debug(f"Error processing {psg_file}: {e}", "ERROR")
                continue
        
        # Second pass: group consecutive epochs into samples
        print_debug(f"Found {len(all_epochs)} individual epochs", "INFO")
        print_debug(f"Creating samples of {self.segment_size} consecutive epochs", "INFO")
        
        # Group by file and sort by start sample
        epochs_by_file = defaultdict(list)
        for psg_file, start_sample, label in all_epochs:
            epochs_by_file[psg_file].append((start_sample, label))
        
        for psg_file, epochs in epochs_by_file.items():
            epochs.sort(key=lambda x: x[0])
            sfreq = self.sfreq_cache[psg_file]
            epoch_samples = int(self.epoch_seconds * sfreq)
            
            # Create sequences of consecutive epochs
            for i in range(len(epochs) - self.segment_size + 1):
                sequence = epochs[i:i + self.segment_size]
                
                # Check if epochs are consecutive (within tolerance)
                is_consecutive = True
                for j in range(1, len(sequence)):
                    expected_start = sequence[j-1][0] + epoch_samples
                    actual_start = sequence[j][0]
                    if abs(expected_start - actual_start) > epoch_samples // 2:
                        is_consecutive = False
                        break
                
                if is_consecutive:
                    start_samples = [s[0] for s in sequence]
                    labels = [s[1] for s in sequence]
                    self.samples.append((psg_file, start_samples, labels))
        
        print_debug(f"Created {len(self.samples)} valid samples", "SUCCESS")
        self._balance_samples()

    def _balance_samples(self):
        """Downsample classes so each has equal number of segments"""
        if not self.samples:
            return

        print_debug("Balancing class distribution across samples...", "INFO")

        sample_entries = []
        class_segment_counts = defaultdict(int)

        for sample in self.samples:
            _, _, labels = sample
            label_counts = defaultdict(int)
            for label in labels:
                label_counts[label] += 1
                class_segment_counts[label] += 1
            sample_entries.append((*sample, label_counts))

        if not class_segment_counts:
            return

        target_count = min(class_segment_counts.values())
        print_debug(f"Target segments per class: {target_count}", "INFO")

        target_per_class = {label: target_count for label in class_segment_counts}
        random.shuffle(sample_entries)

        balanced_samples = []
        running_counts = defaultdict(int)

        for entry in sample_entries:
            psg_file, start_samples, labels, label_counts = entry

            can_add = True
            for label, count in label_counts.items():
                if running_counts[label] + count > target_per_class[label]:
                    can_add = False
                    break

            if not can_add:
                continue

            balanced_samples.append((psg_file, start_samples, labels))
            for label, count in label_counts.items():
                running_counts[label] += count

            if all(running_counts[label] >= target_per_class[label] for label in target_per_class):
                break

        self.samples = balanced_samples

        for label, count in running_counts.items():
            print_debug(f"Class {label}: {count} segments", "INFO")

        print_debug(f"Balanced samples: {len(self.samples)}", "SUCCESS")
    
    def _compute_spectrogram(self, signal_data: np.ndarray, fs: int) -> np.ndarray:
        """Compute spectrogram for a signal with fixed frequency resolution"""
        # Compute spectrogram with fixed parameters
        # All signals are resampled to target_fs=100, so frequency resolution is consistent
        f, t, Sxx = spectrogram(
            signal_data, 
            fs=fs, 
            nperseg=self.nfft, 
            noverlap=self.overlap,
            mode='magnitude'
        )
        
        # Fixed number of frequency bins: nfft//2 + 1 (standard for real signals)
        # But we'll limit to 0-30 Hz for sleep staging relevance
        total_freq_bins = self.nfft // 2 + 1
        freq_resolution = fs / self.nfft  # Frequency resolution in Hz
        max_freq_idx = int(30.0 / freq_resolution) + 1  # Index for 30 Hz
        max_freq_idx = min(max_freq_idx, total_freq_bins)
        
        # Extract frequencies up to 30 Hz
        Sxx = Sxx[:max_freq_idx, :]
        
        # Ensure consistent size by using a fixed number of bins
        # Use the maximum possible number of bins for 0-30 Hz at target_fs
        # This ensures all spectrograms have the same shape
        target_freq_bins = int(30.0 * self.nfft / self.target_fs) + 1
        target_freq_bins = min(target_freq_bins, total_freq_bins)
        
        # Pad or truncate to ensure consistent dimensions
        if Sxx.shape[0] < target_freq_bins:
            # Pad with very small values (not zeros to avoid issues)
            padding = np.full((target_freq_bins - Sxx.shape[0], Sxx.shape[1]), 1e-10)
            Sxx = np.vstack([Sxx, padding])
        elif Sxx.shape[0] > target_freq_bins:
            # Truncate to target size
            Sxx = Sxx[:target_freq_bins, :]
        
        # Log scale and transpose: (time_steps, freq_bins)
        Sxx = np.log1p(Sxx + 1e-10)  # Add small epsilon to avoid log(0)
        return Sxx.T  # (time_steps, freq_bins)
    
    def _load_epoch(self, psg_file: str, start_sample: int) -> np.ndarray:
        """Load a single epoch from a PSG file"""
        psg_path = os.path.join(self.folder_path, psg_file)
        sfreq = self.sfreq_cache[psg_file]
        epoch_samples = int(self.epoch_seconds * sfreq)
        target_samples = int(self.epoch_seconds * self.target_fs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
            
            # Pick available channels
            available_channels = [ch for ch in self.channels if ch in raw.ch_names]
            if len(available_channels) == 0:
                available_channels = raw.ch_names[:min(len(raw.ch_names), 7)]
            
            raw.pick(available_channels)
            
            # Load data
            if start_sample + epoch_samples > raw.n_times:
                data, _ = raw[:, start_sample:]
                padding_size = epoch_samples - data.shape[1]
                if padding_size > 0:
                    padding = np.zeros((data.shape[0], padding_size))
                    data = np.concatenate([data, padding], axis=1)
            else:
                data, _ = raw[:, start_sample:start_sample + epoch_samples]
            
            # Resample if needed
            if sfreq != self.target_fs:
                resampled_data = []
                for ch_data in data:
                    resampled = signal.resample(ch_data, target_samples)
                    resampled_data.append(resampled)
                data = np.array(resampled_data)
            else:
                data = data[:, :target_samples]
        
        return data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample: sequence of epochs with their labels"""
        psg_file, start_samples, labels = self.samples[idx]
        
        epochs_features = []
        
        for start_sample in start_samples:
            # Load epoch data
            epoch_data = self._load_epoch(psg_file, start_sample)
            # epoch_data shape: (channels, time_samples)
            
            if self.use_spectrogram:
                # Compute spectrogram for each channel and concatenate
                channel_features = []
                for ch_data in epoch_data:
                    spec = self._compute_spectrogram(ch_data, self.target_fs)
                    channel_features.append(spec)
                # Concatenate all channel spectrograms
                features = np.concatenate(channel_features, axis=1)
                # features shape: (time_steps, freq_bins * channels)
            else:
                # Use raw signal (transpose to time_steps, channels)
                features = epoch_data.T
            
            epochs_features.append(features)
        
        # Stack epochs: (segment_size, time_steps, feature_dim)
        features_array = np.stack(epochs_features, axis=0)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_array)
        labels_tensor = torch.LongTensor(labels)
        
        return features_tensor, labels_tensor


class CFSSleepStagingDataset(Dataset):
    """
    Dataset for loading and processing CFS (Chronic Fatigue Syndrome) database files.
    Each sample consists of a sequence of 30-second epochs (segments).
    Skips initial wake stages at the beginning of recordings.
    """
    
    def __init__(
        self,
        edf_folder_path: str,
        annotation_folder_path: str,
        segment_size: int = 10,  # Number of consecutive epochs per sample
        epoch_seconds: int = 30,
        target_fs: int = 100,
        use_spectrogram: bool = True,
        nfft: int = 256,
        overlap: int = 128,
        channels: Optional[List[str]] = None,
        filter_unscored: bool = True,
        skip_initial_wake: bool = True
    ):
        self.edf_folder_path = edf_folder_path
        self.annotation_folder_path = annotation_folder_path
        self.segment_size = segment_size
        self.epoch_seconds = epoch_seconds
        self.target_fs = target_fs
        self.use_spectrogram = use_spectrogram
        self.nfft = nfft
        self.overlap = overlap
        self.filter_unscored = filter_unscored
        self.skip_initial_wake = skip_initial_wake
        
        # CFS Sleep stage mapping (Wake is excluded, so we have 5 classes)
        # Original CFS mapping: Wake|0=0, Stage 1|1=1, Stage 2|2=2, Stage 3|3=3, Stage 4|4=4, REM|5=5, Unscored|9=6
        # New mapping (skipping wake): R=0, Stage 1=1, Stage 2=2, Stage 3=3, Stage 4=4
        self.original_stage_mapping = {
            "Wake|0": 0,
            "Stage 1 sleep|1": 1,
            "Stage 2 sleep|2": 2,
            "Stage 3 sleep|3": 3,
            "Stage 4 sleep|4": 4,
            "REM sleep|5": 5,
            "Unscored|9": 6
        }
        
        # Remapped stages (excluding wake): R=0, 1=1, 2=2, 3=3, 4=4
        self.stage_mapping = {
            "REM sleep|5": 0,
            "Stage 1 sleep|1": 1,
            "Stage 2 sleep|2": 2,
            "Stage 3 sleep|3": 3,
            "Stage 4 sleep|4": 4,
            "Unscored|9": 6  # Will be filtered if filter_unscored is True
        }
        self.num_classes = 5  # R, 1, 2, 3, 4 (no wake)
        
        # Default channels - adjust based on CFS data
        self.channels = channels or None  # Will auto-detect if None
        
        # Prepare data index
        self.samples = []
        self.sfreq_cache = {}
        self._prepare_index()
        
        print_debug(f"Dataset initialized with {len(self.samples)} samples", "INFO")
        print_debug(f"Number of classes: {self.num_classes} (R, 1, 2, 3, 4)", "INFO")
    
    def _parse_xml_annotations(self, xml_path: str) -> Optional[mne.Annotations]:
        """Parse XML annotation file and return MNE Annotations"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            onsets = []
            durations = []
            descriptions = []
            
            # Loop through events in XML
            for event in root.findall('.//ScoredEvent'):
                start = float(event.find('Start').text) if event.find('Start') is not None else None
                duration = float(event.find('Duration').text) if event.find('Duration') is not None else None
                event_type = event.find('EventType').text if event.find('EventType') is not None else None
                
                # Only keep sleep stage events
                if event_type is not None and "Stages|Stages" in event_type:
                    stage = event.find('EventConcept').text
                    if stage in self.original_stage_mapping:
                        onsets.append(start)
                        durations.append(duration)
                        descriptions.append(stage)
            
            if not onsets:
                return None
            
            # Convert to MNE Annotations
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            return annotations
        except Exception as e:
            print_debug(f"Error parsing XML {xml_path}: {e}", "ERROR")
            return None
    
    def _find_annotation_file(self, edf_filename: str) -> Optional[str]:
        """Find corresponding XML annotation file for an EDF file"""
        # CFS files typically have format: cfs-visit5-{nsrrid}.edf
        # XML files: {nsrrid}-nsrr.xml
        base_name = Path(edf_filename).stem
        if 'cfs-visit5-' in base_name:
            nsrrid = base_name.replace('cfs-visit5-', '')
            xml_filename = f"{nsrrid}-nsrr.xml"
            xml_path = os.path.join(self.annotation_folder_path, xml_filename)
            if os.path.exists(xml_path):
                return xml_path
        
        # Try alternative pattern: look for any XML file that might match
        for f in os.listdir(self.annotation_folder_path):
            if f.endswith('.xml') and base_name[:17] in f:
                return os.path.join(self.annotation_folder_path, f)
        
        return None
    
    def _prepare_index(self):
        """Prepare index of all valid samples (sequences of consecutive epochs)"""
        print_debug("Preparing CFS dataset index...", "INFO")
        
        # First pass: collect all epochs with their file, start sample, and label
        all_epochs = []
        
        for edf_file in os.listdir(self.edf_folder_path):
            if not edf_file.endswith('.edf'):
                continue
            
            edf_path = os.path.join(self.edf_folder_path, edf_file)
            xml_path = self._find_annotation_file(edf_file)
            
            if not xml_path:
                print_debug(f"No annotation file for {edf_file}, skipping", "WARNING")
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                    sfreq = int(raw.info['sfreq'])
                    self.sfreq_cache[edf_file] = sfreq
                    
                    annotations = self._parse_xml_annotations(xml_path)
                    if annotations is None:
                        print_debug(f"No valid annotations in {xml_path}, skipping", "WARNING")
                        continue
                    
                    epoch_samples = int(self.epoch_seconds * sfreq)
                    
                    # Track epochs for this file
                    file_epochs = []
                    
                    for desc, onset, duration in zip(
                        annotations.description, 
                        annotations.onset, 
                        annotations.duration
                    ):
                        original_label = self.original_stage_mapping.get(desc, 6)
                        
                        # Skip wake stages
                        if original_label == 0:  # Wake
                            continue
                        
                        # Filter unscored if requested
                        if self.filter_unscored and original_label == 6:
                            continue
                        
                        # Map to new label (excluding wake)
                        if desc in self.stage_mapping:
                            label = self.stage_mapping[desc]
                        else:
                            continue
                        
                        full_epochs = int(duration // self.epoch_seconds)
                        start_sample = int(onset * sfreq)
                        
                        for i in range(full_epochs):
                            epoch_start = start_sample + i * epoch_samples
                            epoch_end = epoch_start + epoch_samples
                            
                            if epoch_end <= raw.n_times:
                                file_epochs.append((epoch_start, label))
                    
                    # Skip initial wake stages: find first non-wake epoch
                    if self.skip_initial_wake and file_epochs:
                        file_epochs.sort(key=lambda x: x[0])
                        # All wake stages are already filtered, so we can use all epochs
                        # But we want to skip initial wake if there are any at the start
                        # Since we already filtered wake, we just need to ensure we start from sleep onset
                        all_epochs.extend([(edf_file, start_sample, label) for start_sample, label in file_epochs])
                    else:
                        all_epochs.extend([(edf_file, start_sample, label) for start_sample, label in file_epochs])
                    
            except Exception as e:
                print_debug(f"Error processing {edf_file}: {e}", "ERROR")
                continue
        
        # Second pass: group consecutive epochs into samples
        print_debug(f"Found {len(all_epochs)} individual epochs", "INFO")
        print_debug(f"Creating samples of {self.segment_size} consecutive epochs", "INFO")
        
        # Group by file and sort by start sample
        epochs_by_file = defaultdict(list)
        for edf_file, start_sample, label in all_epochs:
            epochs_by_file[edf_file].append((start_sample, label))
        
        for edf_file, epochs in epochs_by_file.items():
            epochs.sort(key=lambda x: x[0])
            sfreq = self.sfreq_cache[edf_file]
            epoch_samples = int(self.epoch_seconds * sfreq)
            
            # Create sequences of consecutive epochs
            for i in range(len(epochs) - self.segment_size + 1):
                sequence = epochs[i:i + self.segment_size]
                
                # Check if epochs are consecutive (within tolerance)
                is_consecutive = True
                for j in range(1, len(sequence)):
                    expected_start = sequence[j-1][0] + epoch_samples
                    actual_start = sequence[j][0]
                    if abs(expected_start - actual_start) > epoch_samples // 2:
                        is_consecutive = False
                        break
                
                if is_consecutive:
                    start_samples = [s[0] for s in sequence]
                    labels = [s[1] for s in sequence]
                    self.samples.append((edf_file, start_samples, labels))
        
        print_debug(f"Created {len(self.samples)} valid samples", "SUCCESS")
        self._balance_samples()
    
    def _balance_samples(self):
        """Downsample classes so each has equal number of segments"""
        if not self.samples:
            return

        print_debug("Balancing class distribution across samples...", "INFO")

        sample_entries = []
        class_segment_counts = defaultdict(int)

        for sample in self.samples:
            _, _, labels = sample
            label_counts = defaultdict(int)
            for label in labels:
                label_counts[label] += 1
                class_segment_counts[label] += 1
            sample_entries.append((*sample, label_counts))

        if not class_segment_counts:
            return

        target_count = min(class_segment_counts.values())
        print_debug(f"Target segments per class: {target_count}", "INFO")

        target_per_class = {label: target_count for label in class_segment_counts}
        random.shuffle(sample_entries)

        balanced_samples = []
        running_counts = defaultdict(int)

        for entry in sample_entries:
            edf_file, start_samples, labels, label_counts = entry

            can_add = True
            for label, count in label_counts.items():
                if running_counts[label] + count > target_per_class[label]:
                    can_add = False
                    break

            if not can_add:
                continue

            balanced_samples.append((edf_file, start_samples, labels))
            for label, count in label_counts.items():
                running_counts[label] += count

            if all(running_counts[label] >= target_per_class[label] for label in target_per_class):
                break

        self.samples = balanced_samples

        for label, count in running_counts.items():
            stage_name = ['R', '1', '2', '3', '4'][label] if label < 5 else 'Unknown'
            print_debug(f"Class {stage_name} ({label}): {count} segments", "INFO")

        print_debug(f"Balanced samples: {len(self.samples)}", "SUCCESS")
    
    def _compute_spectrogram(self, signal_data: np.ndarray, fs: int) -> np.ndarray:
        """Compute spectrogram for a signal with fixed frequency resolution"""
        # Compute spectrogram with fixed parameters
        f, t, Sxx = spectrogram(
            signal_data, 
            fs=fs, 
            nperseg=self.nfft, 
            noverlap=self.overlap,
            mode='magnitude'
        )
        
        # Fixed number of frequency bins: nfft//2 + 1 (standard for real signals)
        # But we'll limit to 0-30 Hz for sleep staging relevance
        total_freq_bins = self.nfft // 2 + 1
        freq_resolution = fs / self.nfft  # Frequency resolution in Hz
        max_freq_idx = int(30.0 / freq_resolution) + 1  # Index for 30 Hz
        max_freq_idx = min(max_freq_idx, total_freq_bins)
        
        # Extract frequencies up to 30 Hz
        Sxx = Sxx[:max_freq_idx, :]
        
        # Ensure consistent size by using a fixed number of bins
        target_freq_bins = int(30.0 * self.nfft / self.target_fs) + 1
        target_freq_bins = min(target_freq_bins, total_freq_bins)
        
        # Pad or truncate to ensure consistent dimensions
        if Sxx.shape[0] < target_freq_bins:
            # Pad with very small values (not zeros to avoid issues)
            padding = np.full((target_freq_bins - Sxx.shape[0], Sxx.shape[1]), 1e-10)
            Sxx = np.vstack([Sxx, padding])
        elif Sxx.shape[0] > target_freq_bins:
            # Truncate to target size
            Sxx = Sxx[:target_freq_bins, :]
        
        # Log scale and transpose: (time_steps, freq_bins)
        Sxx = np.log1p(Sxx + 1e-10)  # Add small epsilon to avoid log(0)
        return Sxx.T  # (time_steps, freq_bins)
    
    def _load_epoch(self, edf_file: str, start_sample: int) -> np.ndarray:
        """Load a single epoch from an EDF file"""
        edf_path = os.path.join(self.edf_folder_path, edf_file)
        sfreq = self.sfreq_cache[edf_file]
        epoch_samples = int(self.epoch_seconds * sfreq)
        target_samples = int(self.epoch_seconds * self.target_fs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            
            # Pick available channels
            if self.channels:
                available_channels = [ch for ch in self.channels if ch in raw.ch_names]
            else:
                # Auto-detect: prefer EEG channels
                available_channels = [ch for ch in raw.ch_names if 'EEG' in ch or 'EEG' in ch.upper()]
                if len(available_channels) == 0:
                    available_channels = raw.ch_names[:min(len(raw.ch_names), 7)]
                else:
                    available_channels = available_channels[:min(len(available_channels), 7)]
            
            if len(available_channels) == 0:
                available_channels = raw.ch_names[:min(len(raw.ch_names), 7)]
            
            raw.pick(available_channels)
            
            # Load data
            if start_sample + epoch_samples > raw.n_times:
                data, _ = raw[:, start_sample:]
                padding_size = epoch_samples - data.shape[1]
                if padding_size > 0:
                    padding = np.zeros((data.shape[0], padding_size))
                    data = np.concatenate([data, padding], axis=1)
            else:
                data, _ = raw[:, start_sample:start_sample + epoch_samples]
            
            # Resample if needed
            if sfreq != self.target_fs:
                resampled_data = []
                for ch_data in data:
                    resampled = signal.resample(ch_data, target_samples)
                    resampled_data.append(resampled)
                data = np.array(resampled_data)
            else:
                data = data[:, :target_samples]
        
        return data
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample: sequence of epochs with their labels"""
        edf_file, start_samples, labels = self.samples[idx]
        
        epochs_features = []
        
        for start_sample in start_samples:
            # Load epoch data
            epoch_data = self._load_epoch(edf_file, start_sample)
            # epoch_data shape: (channels, time_samples)
            
            if self.use_spectrogram:
                # Compute spectrogram for each channel and concatenate
                channel_features = []
                for ch_data in epoch_data:
                    spec = self._compute_spectrogram(ch_data, self.target_fs)
                    channel_features.append(spec)
                # Concatenate all channel spectrograms
                features = np.concatenate(channel_features, axis=1)
                # features shape: (time_steps, freq_bins * channels)
            else:
                # Use raw signal (transpose to time_steps, channels)
                features = epoch_data.T
            
            epochs_features.append(features)
        
        # Stack epochs: (segment_size, time_steps, feature_dim)
        features_array = np.stack(epochs_features, axis=0)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_array)
        labels_tensor = torch.LongTensor(labels)
        
        return features_tensor, labels_tensor


# ============================================================================
# Training and Evaluation
# ============================================================================

class CheckpointManager:
    """Manages model checkpoints: saving, loading, and tracking best model"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
        metadata: Optional[Dict] = None
    ):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'metadata': metadata or {}
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print_debug(f"Saved checkpoint: {checkpoint_path}", "INFO")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            print_debug(f"Saved best model: {best_path} (val_acc={val_acc:.4f})", "SUCCESS")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer, scheduler=None):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        print_debug(f"Loaded checkpoint from epoch {checkpoint['epoch']}", "SUCCESS")
        print_debug(f"Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch})", "INFO")
        
        return checkpoint['epoch'], checkpoint.get('metadata', {})


def train_epoch(model, dataloader, optimizer, criterion, device, epoch: int):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)  # (batch, segments, time, features)
        labels = labels.to(device)  # (batch, segments)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)  # (batch, segments, num_classes)
        
        # Reshape for loss computation
        batch_size, num_segments, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        loss = criterion(logits_flat, labels_flat)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits_flat, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels_flat.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            batch_acc = accuracy_score(labels_flat.cpu().numpy(), predictions)
            print_debug(
                f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}",
                "TRAIN"
            )
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, epoch: int):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(features)
            
            # Reshape for loss computation
            batch_size, num_segments, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            loss = criterion(logits_flat, labels_flat)
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits_flat, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels_flat.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Compute per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Print per-class metrics
    class_names = ['R', '1', '2', '3', '4']  # No wake stage
    print_debug("Per-class metrics:", "VAL")
    for i, name in enumerate(class_names):
        if i < len(precision):
            print_debug(
                f"  {name}: Precision={precision[i]:.4f}, "
                f"Recall={recall[i]:.4f}, F1={f1[i]:.4f}",
                "VAL"
            )
    
    return avg_loss, accuracy


def train_model(
    edf_folder_path: str,
    annotation_folder_path: str,
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    segment_size: int = 10,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers_local: int = 4,
    num_encoder_layers_global: int = 2,
    dropout: float = 0.1,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
    device: Optional[str] = None
):
    """Main training function"""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_debug(f"Using device: {device}", "INFO")
    
    # Create datasets
    print_debug("Loading CFS datasets...", "INFO")
    full_dataset = CFSSleepStagingDataset(
        edf_folder_path=edf_folder_path,
        annotation_folder_path=annotation_folder_path,
        segment_size=segment_size,
        use_spectrogram=True,
        filter_unscored=True,
        skip_initial_wake=True
    )
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print_debug(f"Train samples: {len(train_dataset)}", "INFO")
    print_debug(f"Validation samples: {len(val_dataset)}", "INFO")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get input dimension from a sample
    sample_features, _ = train_dataset[0]
    _, time_steps, input_dim = sample_features.shape
    print_debug(f"Input shape: (segments={segment_size}, time_steps={time_steps}, features={input_dim})", "INFO")
    
    # Create model
    print_debug("Creating model...", "INFO")
    model = HierarchicalTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers_local=num_encoder_layers_local,
        num_encoder_layers_global=num_encoder_layers_global,
        num_classes=full_dataset.num_classes,
        dropout=dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print_debug(f"Model parameters: {num_params:,}", "INFO")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=False)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            start_epoch, metadata = checkpoint_manager.load_checkpoint(
                resume_from, model, optimizer, scheduler
            )
            start_epoch += 1
            print_debug(f"Resuming from epoch {start_epoch}", "INFO")
        else:
            print_debug(f"Checkpoint not found: {resume_from}", "WARNING")
    
    # Training loop
    print_debug("Starting training...", "SUCCESS")
    print_colored("=" * 80, Colors.BOLD)
    
    for epoch in range(start_epoch, num_epochs):
        print_colored(f"\nEpoch {epoch+1}/{num_epochs}", Colors.BOLD + Colors.HEADER)
        print_colored("-" * 80, Colors.GRAY)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch+1
        )
        print_debug(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", "TRAIN")
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch+1
        )
        print_debug(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", "VAL")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print_debug(f"Learning rate reduced: {old_lr:.6f} -> {current_lr:.6f}", "INFO")
        else:
            print_debug(f"Learning rate: {current_lr:.6f}", "INFO")
        
        # Save checkpoint
        is_best = val_acc > checkpoint_manager.best_val_acc
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            is_best=is_best,
            metadata={
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'num_encoder_layers_local': num_encoder_layers_local,
                'num_encoder_layers_global': num_encoder_layers_global,
                'num_classes': full_dataset.num_classes,
                'segment_size': segment_size
            }
        )
        
        print_colored("-" * 80, Colors.GRAY)
    
    print_colored("=" * 80, Colors.BOLD)
    print_debug("Training completed!", "SUCCESS")
    print_debug(f"Best validation accuracy: {checkpoint_manager.best_val_acc:.4f} (epoch {checkpoint_manager.best_epoch})", "SUCCESS")
    
    return model, checkpoint_manager


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Sleep Staging Transformer")
    parser.add_argument(
        "--edf_folder_path",
        type=str,
        default=r"D:\cfs\polysomnography\edfs",
        help="Path to CFS EDF files folder"
    )
    parser.add_argument(
        "--annotation_folder_path",
        type=str,
        default=r"D:\cfs\polysomnography\annotations-events-nsrr",
        help="Path to CFS XML annotation files folder"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--segment_size", type=int, default=10, help="Number of consecutive epochs per sample")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers_local", type=int, default=4, help="Local encoder layers")
    parser.add_argument("--num_encoder_layers_global", type=int, default=2, help="Global encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Train model
    model, checkpoint_manager = train_model(
        edf_folder_path=args.edf_folder_path,
        annotation_folder_path=args.annotation_folder_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        segment_size=args.segment_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_encoder_layers_local=args.num_encoder_layers_local,
        num_encoder_layers_global=args.num_encoder_layers_global,
        dropout=args.dropout,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from
    )

