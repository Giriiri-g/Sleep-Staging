"""
Sleep Staging Transformer - Full Implementation (Updated)
===============================================

A hierarchical transformer model for sleep stage classification using PSG data from EDF files.
Includes data loading, feature extraction, training, validation, and checkpointing.
Updated with enhanced preprocessing based on SOTA literature (e.g., SleepTransformer, IITNet).

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
from typing import Tuple, Optional, Dict, List, Set
import json
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy import signal
from scipy.signal import stft
from scipy.interpolate import interp1d

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
        # x: (batch, segments, time_steps, channels, freq_bins)
        batch_size, num_segments, time_steps, channels, freq_bins = x.shape

        # Flatten channel × frequency → input_dim
        x = x.view(batch_size, num_segments, time_steps, channels * freq_bins)

        # Prepare for local encoder
        # reshape: (batch * segments, time_steps, input_dim)
        x = x.view(batch_size * num_segments, time_steps, channels * freq_bins)

        # Local encoder
        local_embeddings = self.local_encoder(x)

        # Reshape back: (batch, segments, hidden_dim)
        local_embeddings = local_embeddings.view(batch_size, num_segments, -1)

        # Global encoder
        global_embeddings = self.global_encoder(local_embeddings)

        # Prediction per segment
        predictions = self.prediction_head(global_embeddings)

        return predictions


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================


class CFSSleepStagingDataset(Dataset):
    """
    Dataset for loading and processing CFS (Chronic Fatigue Syndrome) database files.
    Each sample consists of a sequence of 30-second epochs (segments).
    Skips initial wake stages at the beginning of recordings.
    Updated with enhanced spectrogram computation based on SOTA literature.
    """
    
    def __init__(
        self,
        edf_folder_path: str,
        annotation_folder_path: str,
        segment_size: int = 10,  # Number of consecutive epochs per sample
        epoch_seconds: int = 30,
        target_fs: int = 100,
        use_spectrogram: bool = True,
        nfft: int = 1024,  # Updated: Higher resolution
        overlap: int = 924,  # Updated: For 1s hop (100 samples at 100Hz)
        channels: Optional[List[str]] = None,
        filter_unscored: bool = True,
        skip_initial_wake: bool = True,
        balance_classes: bool = False,
        apply_channel_normalization: bool = True,
        feature_normalization: bool = True,
        spec_augment_prob: float = 0.0,
        time_mask_param: int = 10,
        freq_mask_param: int = 10,
        max_time_masks: int = 2,
        max_freq_masks: int = 2
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
        self.balance_classes = balance_classes
        self.apply_channel_normalization = apply_channel_normalization
        self.feature_normalization = feature_normalization
        self.spec_augment_prob = spec_augment_prob
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.max_time_masks = max_time_masks
        self.max_freq_masks = max_freq_masks
        self.augment_indices: Set[int] = set()
        
        # Original CFS stage mapping: keep full mapping (including Wake)
        # Keys are the annotation strings found in XML -> integer labels
        self.original_stage_mapping = {
            "Wake|0": 0,
            "Stage 1 sleep|1": 1,
            "Stage 2 sleep|2": 2,
            "Stage 3 sleep|3": 3,
            "Stage 4 sleep|4": 4,
            "REM sleep|5": 5
        }

        # Use the original mapping for labels by default. We will only optionally
        # clip initial wake epochs (skip them at the start) but keep Wake label (0)
        # for epochs occurring after sleep onset.
        self.stage_mapping = self.original_stage_mapping.copy()

        # Number of classes: keep full 6-class space (0..5)
        self.num_classes = 6

        # Human-readable class names in label order (0..5)
        self.class_names = ['W', '1', '2', '3', '4', 'R']
        
        # Standard channels for sleep staging (updated to use differential/bipolar)
        self.channels = channels or [
            'EEG C3-M2', 'EEG C4-M1', 'EOG LOC-M2', 'EOG ROC-M1', 'EMG Chin'
        ]  # Updated: Standard PSG channels; will check availability
        
        # Prepare data index
        self.samples = []
        self.sfreq_cache = {}
        self._prepare_index()
        
        print_debug(f"Dataset initialized with {len(self.samples)} samples", "INFO")
        print_debug(f"Number of classes: {self.num_classes} (W, 1, 2, 3, 4, R)", "INFO")
    
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
                    
                    initial_wake_epochs = 0
                    sleep_started = False
                    
                    for desc, onset, duration in zip(
                        annotations.description,
                        annotations.onset,
                        annotations.duration
                    ):
                        original_label = self.original_stage_mapping.get(desc, 6)
                        full_epochs = int(duration // self.epoch_seconds)
                        start_sample = int(onset * sfreq)

                        # Filter unscored if requested
                        if self.filter_unscored and original_label == 6:
                            continue

                        # Handle wake epochs:
                        # - If skipping initial wake and we haven't seen sleep yet, count
                        #   these epochs as clipped and do not include them.
                        # - Otherwise include wake epochs with label 0.
                        if original_label == 0:
                            if self.skip_initial_wake and not sleep_started:
                                initial_wake_epochs += full_epochs
                                continue
                            label = 0
                            # do not set sleep_started = True for wake
                        else:
                            # Non-wake stage: keep original label and mark sleep started
                            label = original_label
                            sleep_started = True

                        for i in range(full_epochs):
                            epoch_start = start_sample + i * epoch_samples
                            epoch_end = epoch_start + epoch_samples

                            if epoch_end <= raw.n_times:
                                file_epochs.append((epoch_start, label))
                    
                    if self.skip_initial_wake:
                        if initial_wake_epochs > 0:
                            print_debug(
                                f"Clipped {initial_wake_epochs} initial wake epochs for {edf_file}",
                                "INFO"
                            )
                        else:
                            print_debug(
                                f"No initial wake epochs to clip for {edf_file}",
                                "INFO"
                            )
                    
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
        if self.balance_classes:
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
            stage_name = ['W', '1', '2', '3', '4', 'R', 'U'][label]
            print_debug(f"Class {stage_name} ({label}): {count} segments", "INFO")

        print_debug(f"Balanced samples: {len(self.samples)}", "SUCCESS")
    
    def _compute_spectrogram(self, signal_data: np.ndarray, fs: int) -> np.ndarray:
        """Enhanced spectrogram computation: high-res STFT, dB scale, fixed freq grid, robust scaling"""
        n_channels = signal_data.shape[0]
        features = []
        
        target_freqs = np.linspace(0.5, 40.0, 128)  # Fixed: 128 bins from 0.5-40 Hz
        
        for ch_idx in range(n_channels):
            x = signal_data[ch_idx]
            
            # High-res STFT (nfft=1024, hop=100 for 1s steps at 100Hz)
            f, t, Zxx = stft(
                x, fs=fs, nperseg=self.nfft, noverlap=self.overlap,
                window='hann', boundary=None, padded=False
            )
            power = np.abs(Zxx)**2
            log_power = 10 * np.log10(power + 1e-12)  # dB scale
            
            # Interpolate to fixed frequency grid
            interp_func = interp1d(f, log_power, axis=0, kind='linear',
                                   bounds_error=False, fill_value=-10)
            log_power_fixed = interp_func(target_freqs)  # (128, time_bins)
            
            features.append(log_power_fixed)
        
        spec = np.stack(features, axis=0)  # (n_channels, 128, time_bins)
        
        # Robust scaling (median/IQR) per channel
        median = np.median(spec, axis=(1, 2), keepdims=True)
        q75 = np.percentile(spec, 75, axis=(1, 2), keepdims=True)
        q25 = np.percentile(spec, 25, axis=(1, 2), keepdims=True)
        spec = (spec - median) / (q75 - q25 + 1e-8)
        
        return spec.transpose(2, 0, 1)  # (time_bins, n_channels, 128) for model input
    
    def _load_epoch(self, edf_file: str, start_sample: int) -> np.ndarray:
        """Load a single epoch from an EDF file"""
        edf_path = os.path.join(self.edf_folder_path, edf_file)
        sfreq = self.sfreq_cache[edf_file]
        epoch_samples = int(self.epoch_seconds * sfreq)
        target_samples = int(self.epoch_seconds * self.target_fs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            
            # Pick available channels (prefer standard PSG channels)
            available_channels = [ch for ch in self.channels if ch in raw.ch_names]
            if not available_channels:
                available_channels = raw.ch_names[:min(len(raw.ch_names), len(self.channels))]
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
            if self.apply_channel_normalization:
                epoch_data = self._normalize_epoch_signal(epoch_data)
            
            if self.use_spectrogram:
                # Compute enhanced spectrogram
                features = self._compute_spectrogram(epoch_data, self.target_fs)
                # features shape: (time_steps, channels, freq_bins)
            else:
                # Use raw signal (transpose to time_steps, channels)
                features = epoch_data.T
            
            epochs_features.append(features)
        
        # Stack epochs: (segment_size, time_steps, channels, freq_bins)
        features_array = np.stack(epochs_features, axis=0)
        if self.feature_normalization:
            features_array = self._normalize_features(features_array)
        if self._should_augment(idx):
            features_array = self._apply_spec_augment(features_array)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_array)
        labels_tensor = torch.LongTensor(labels)
        
        return features_tensor, labels_tensor

    def set_augmentation_indices(self, indices: List[int]):
        """Mark dataset indices eligible for data augmentation."""
        self.augment_indices = set(indices)

    def _should_augment(self, idx: int) -> bool:
        if self.spec_augment_prob <= 0.0 or not self.augment_indices:
            return False
        if idx not in self.augment_indices:
            return False
        return random.random() < self.spec_augment_prob

    def _normalize_epoch_signal(self, epoch_data: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization."""
        mean = epoch_data.mean(axis=1, keepdims=True)
        std = epoch_data.std(axis=1, keepdims=True) + 1e-8
        return (epoch_data - mean) / std

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize spectrogram features per sample."""
        mean = features.mean(axis=(0, 1, 2), keepdims=True)
        std = features.std(axis=(0, 1, 2), keepdims=True) + 1e-6
        return (features - mean) / std

    def _apply_spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Apply SpecAugment-style time and frequency masking (adapted for 3D input)."""
        augmented = features.copy()
        num_epochs, time_steps, n_channels, freq_bins = augmented.shape
        for epoch_idx in range(num_epochs):
            for ch_idx in range(n_channels):
                epoch_feat = augmented[epoch_idx, :, ch_idx, :]  # (time, freq)

                # Time masks
                for _ in range(self.max_time_masks):
                    t = random.randint(0, max(0, self.time_mask_param))
                    if t == 0 or t >= time_steps:
                        continue
                    t0 = random.randint(0, max(0, time_steps - t))
                    epoch_feat[t0:t0 + t, :] = 0.0

                # Frequency masks
                for _ in range(self.max_freq_masks):
                    f = random.randint(0, max(0, self.freq_mask_param))
                    if f == 0 or f >= freq_bins:
                        continue
                    f0 = random.randint(0, max(0, freq_bins - f))
                    epoch_feat[:, f0:f0 + f] = 0.0

                augmented[epoch_idx, :, ch_idx, :] = epoch_feat
        return augmented


def compute_class_distribution(dataset: CFSSleepStagingDataset, subset_indices: List[int]) -> torch.Tensor:
    """Count per-class segment occurrences for a subset of samples."""
    counts = torch.zeros(dataset.num_classes, dtype=torch.long)
    for idx in subset_indices:
        _, _, labels = dataset.samples[idx]
        for label in labels:
            if 0 <= label < dataset.num_classes:
                counts[label] += 1
    return counts


def compute_class_weights(class_counts: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Convert class counts into normalized inverse-frequency weights."""
    counts = class_counts.float().clamp_min(epsilon)
    total = counts.sum()
    num_classes = counts.numel()
    weights = total / (counts * num_classes)
    # Normalize to keep average weight == 1 for stable loss scaling
    return weights / weights.mean()


def build_sample_weights(dataset: CFSSleepStagingDataset, subset_indices: List[int], class_weights: torch.Tensor) -> torch.DoubleTensor:
    """Create per-sample weights by averaging constituent label weights."""
    weights = []
    for idx in subset_indices:
        _, _, labels = dataset.samples[idx]
        if labels:
            label_tensor = torch.tensor(labels, dtype=torch.long)
            label_weights = class_weights[label_tensor].mean()
        else:
            label_weights = class_weights.mean()
        weights.append(label_weights.item())
    return torch.DoubleTensor(weights)


def get_subset_indices(subset) -> List[int]:
    """Return the indices that back a dataset or subset."""
    if isinstance(subset, Subset):
        return subset.indices
    return list(range(len(subset)))


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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch: int, scheduler=None, scheduler_step_per_batch: bool = False, confidence_penalty: float = 0.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)  # (batch, segments, time, channels, freq)
        labels = labels.to(device)  # (batch, segments)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)  # (batch, segments, num_classes)
        
        # Reshape for loss computation
        batch_size, num_segments, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        ce_loss = criterion(logits_flat, labels_flat)
        if confidence_penalty > 0.0:
            log_probs = F.log_softmax(logits_flat, dim=1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
            loss = ce_loss - confidence_penalty * entropy
        else:
            loss = ce_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler and scheduler_step_per_batch:
            scheduler.step()
        
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
    # Determine class names from dataloader.dataset if available, else default to 6-class mapping
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'class_names'):
        class_names = dataloader.dataset.class_names
    else:
        class_names = ['W', '1', '2', '3', '4', 'R']

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
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    segment_size: int = 10,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers_local: int = 4,
    num_encoder_layers_global: int = 2,
    dropout: float = 0.1,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
    device: Optional[str] = None,
    balance_classes: bool = False,
    label_smoothing: float = 0.05,
    confidence_penalty: float = 0.02,
    scheduler_type: str = "one_cycle",
    spec_augment_prob: float = 0.5,
    time_mask_param: int = 12,
    freq_mask_param: int = 24,
    max_time_masks: int = 2,
    max_freq_masks: int = 2,
    apply_channel_normalization: bool = True,
    feature_normalization: bool = True
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
        skip_initial_wake=True,
        balance_classes=balance_classes,
        apply_channel_normalization=apply_channel_normalization,
        feature_normalization=feature_normalization,
        spec_augment_prob=spec_augment_prob,
        time_mask_param=time_mask_param,
        freq_mask_param=freq_mask_param,
        max_time_masks=max_time_masks,
        max_freq_masks=max_freq_masks
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

    train_indices = get_subset_indices(train_dataset)
    val_indices = get_subset_indices(val_dataset)

    train_class_counts = compute_class_distribution(full_dataset, train_indices)
    val_class_counts = compute_class_distribution(full_dataset, val_indices)

    def log_distribution(name: str, counts: torch.Tensor):
        total = counts.sum().item()
        if total == 0:
            print_debug(f"{name} has zero labeled segments", "WARNING")
            return
        print_debug(f"{name} class distribution (segments):", "INFO")
        for i, count in enumerate(counts):
            class_name = full_dataset.class_names[i] if i < len(full_dataset.class_names) else f"Class {i}"
            pct = (count.item() / total) * 100
            print_debug(f"  {class_name}: {count.item()} ({pct:.2f}%)", "INFO")

    log_distribution("Train", train_class_counts)
    log_distribution("Validation", val_class_counts)

    full_dataset.set_augmentation_indices(train_indices)

    class_weights = compute_class_weights(train_class_counts)
    print_debug(f"Using class weights: {class_weights.tolist()}", "INFO")

    train_sampler = None
    if len(train_indices) > 0:
        sample_weights = build_sample_weights(full_dataset, train_indices, class_weights)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    _, time_steps, n_channels, freq_bins = sample_features.shape
    input_dim = n_channels * freq_bins  # Flatten channels and freq for transformer input
    print_debug(f"Input shape: (segments={segment_size}, time_steps={time_steps}, input_dim={input_dim})", "INFO")
    
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
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = None
    scheduler_step_per_batch = False
    scheduler_name = (scheduler_type or "one_cycle").lower()
    steps_per_epoch = len(train_loader)

    if scheduler_name == "one_cycle":
        if steps_per_epoch == 0:
            print_debug("OneCycleLR skipped because training loader is empty", "WARNING")
            scheduler = None
        else:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=num_epochs,
                steps_per_epoch=max(1, steps_per_epoch),
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1000.0,
                anneal_strategy='cos'
            )
            scheduler_step_per_batch = True
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    else:
        scheduler_name = "plateau"
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
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
            model, train_loader, optimizer, criterion, device, epoch+1,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
            confidence_penalty=confidence_penalty
        )
        print_debug(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}", "TRAIN")
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch+1
        )
        print_debug(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", "VAL")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler and not scheduler_step_per_batch:
            if scheduler_name == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print_debug(f"Learning rate updated: {old_lr:.6f} -> {current_lr:.6f}", "INFO")
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
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-2, help="Learning rate")
    parser.add_argument("--segment_size", type=int, default=10, help="Number of consecutive epochs per sample")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers_local", type=int, default=4, help="Local encoder layers")
    parser.add_argument("--num_encoder_layers_global", type=int, default=2, help="Global encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Enable aggressive downsampling to equalize classes (disabled by default)"
    )
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing for cross-entropy")
    parser.add_argument("--confidence_penalty", type=float, default=0.02, help="Entropy-based confidence penalty")
    parser.add_argument("--scheduler_type", type=str, default="one_cycle", choices=["one_cycle", "cosine", "plateau"], help="Learning rate scheduler strategy")
    parser.add_argument("--spec_augment_prob", type=float, default=0.5, help="Probability of applying SpecAugment to a sample")
    parser.add_argument("--time_mask_param", type=int, default=12, help="Max time mask width for SpecAugment")
    parser.add_argument("--freq_mask_param", type=int, default=24, help="Max frequency mask width for SpecAugment")
    parser.add_argument("--max_time_masks", type=int, default=2, help="Number of time masks for SpecAugment")
    parser.add_argument("--max_freq_masks", type=int, default=2, help="Number of frequency masks for SpecAugment")
    parser.add_argument("--disable_channel_norm", action="store_true", help="Disable per-channel z-score normalization")
    parser.add_argument("--disable_feature_norm", action="store_true", help="Disable feature-level normalization")
    
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
        resume_from=args.resume_from,
        balance_classes=args.balance_classes,
        label_smoothing=args.label_smoothing,
        confidence_penalty=args.confidence_penalty,
        scheduler_type=args.scheduler_type,
        spec_augment_prob=args.spec_augment_prob,
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        max_time_masks=args.max_time_masks,
        max_freq_masks=args.max_freq_masks,
        apply_channel_normalization=not args.disable_channel_norm,
        feature_normalization=not args.disable_feature_norm
    )