"""
Dataset utilities for the CFS Visit 5 cohort.
Reads EDF paths and multi-label ailment targets from a CSV file and returns
fixed-length multivariate sequences for NAS (DARTS/RL) workflows.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import print_info, print_warning, print_key_value

# Suppress MNE noise early
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import mne  # noqa: E402

mne.set_log_level("ERROR")


def _resolve_split(value: float, total: int) -> int:
    """
    Interpret split value as absolute count if >= 1,
    otherwise treat it as ratio of total.
    """
    if value is None or total == 0:
        return 0
    if isinstance(value, (int, np.integer)) or value >= 1.0:
        return int(value)
    return int(total * float(value))


class CFSAilmentDataset(Dataset):
    """Dataset that lazily loads EDF signals for multi-label ailment prediction."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        indices: Sequence[int],
        input_channels: int,
        input_length: int,
        channel_names: Optional[Sequence[str]] = None,
        target_sample_rate: float = 100.0,
        normalization: str = "zscore",
    ):
        self.df = dataframe.iloc[list(indices)].reset_index(drop=True)
        self.input_channels = input_channels
        self.input_length = input_length
        self.target_sample_rate = target_sample_rate
        self.normalization = normalization
        self.channel_names = [ch.strip() for ch in channel_names] if channel_names else None

        self.path_column = "path"
        self.label_columns = [col for col in dataframe.columns if col != self.path_column]
        self.num_labels = len(self.label_columns)

        self.multi_label = True
        self.task_type = "multi_label"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = Path(row[self.path_column])

        # Fast path: preprocessed tensor file (.pt)
        if path.suffix == ".pt":
            if not path.exists():
                raise FileNotFoundError(f"Preprocessed tensor file not found: {path}")
            signal_tensor = torch.load(path, map_location="cpu")
            if not isinstance(signal_tensor, torch.Tensor):
                signal_tensor = torch.tensor(signal_tensor, dtype=torch.float32)
            else:
                signal_tensor = signal_tensor.to(dtype=torch.float32)

        else:
            edf_path = path

            # Debug: trace EDF loading for first few samples
            if idx < 3:
                print_info(f"[CFS] __getitem__ start idx={idx}, path={edf_path}")

            if not edf_path.exists():
                raise FileNotFoundError(f"EDF file not found: {edf_path}")

            t0 = time.time()
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            load_time = time.time() - t0
            if idx < 3:
                print_info(
                    f"[CFS] Loaded EDF idx={idx} in {load_time:.2f}s "
                    f"(sfreq={raw.info.get('sfreq', 'NA')}, n_channels={len(raw.ch_names)})"
                )

            if self.target_sample_rate and abs(raw.info["sfreq"] - self.target_sample_rate) > 1e-3:
                raw.resample(self.target_sample_rate)
                if idx < 3:
                    print_info(
                        f"[CFS] Resampled EDF idx={idx} to {self.target_sample_rate} Hz"
                    )

            signal = self._extract_channels(raw)
            signal = self._normalize(signal)
            signal_tensor = torch.tensor(signal, dtype=torch.float32)
            signal_tensor = self._ensure_length(signal_tensor)

        labels = torch.tensor(row[self.label_columns].values.astype(np.float32))
        labels = torch.where(labels < 0, torch.zeros_like(labels), labels)

        return signal_tensor, labels

    def _extract_channels(self, raw: mne.io.BaseRaw) -> np.ndarray:
        available = raw.ch_names
        if self.channel_names:
            picks = [ch for ch in self.channel_names if ch in available]
        else:
            picks = available[: self.input_channels]

        if not picks:
            raise RuntimeError("No matching channels found in EDF file.")

        data = raw.get_data(picks=picks)
        if data.shape[0] < self.input_channels:
            pad = np.zeros((self.input_channels - data.shape[0], data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
        elif data.shape[0] > self.input_channels:
            data = data[: self.input_channels]

        return data

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True) + 1e-6
            return (signal - mean) / std
        if self.normalization == "minmax":
            min_val = signal.min(axis=1, keepdims=True)
            max_val = signal.max(axis=1, keepdims=True)
            return (signal - min_val) / (max_val - min_val + 1e-6)
        return signal

    def _ensure_length(self, tensor: torch.Tensor) -> torch.Tensor:
        current_len = tensor.shape[1]
        if current_len == self.input_length:
            return tensor
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.input_length, mode="linear", align_corners=False)
        return tensor.squeeze(0)


def load_cfs_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV must include a 'path' column with EDF file paths.")

    df["path"] = df["path"].astype(str)
    exists_mask = df["path"].apply(lambda p: Path(p).exists())
    missing = df.loc[~exists_mask]
    if not missing.empty:
        print_warning(f"Skipping {len(missing)} entries with missing EDF files.")
    df = df.loc[exists_mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("No EDF files found for provided CSV.")
    return df


def create_cfs_dataloaders(
    csv_path: str,
    batch_size: int,
    input_channels: int,
    input_length: int,
    val_split: float,
    test_split: float,
    num_workers: int = 0,
    seed: int = 42,
    channel_names: Optional[Sequence[str]] = None,
    target_sample_rate: float = 100.0,
    normalization: str = "zscore",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], dict]:
    df = load_cfs_dataframe(csv_path)
    total = len(df)

    val_size = min(_resolve_split(val_split, total), total)
    test_size = min(_resolve_split(test_split, total - val_size), total - val_size)
    train_size = total - val_size - test_size

    if train_size <= 0:
        raise ValueError("Not enough samples left for training after val/test splits.")

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    val_indices = indices[:val_size]
    test_indices = indices[val_size : val_size + test_size]
    train_indices = indices[val_size + test_size :]

    dataset_kwargs = dict(
        dataframe=df,
        input_channels=input_channels,
        input_length=input_length,
        channel_names=channel_names,
        target_sample_rate=target_sample_rate,
        normalization=normalization,
    )

    train_dataset = CFSAilmentDataset(indices=train_indices, **dataset_kwargs)
    val_dataset = CFSAilmentDataset(indices=val_indices, **dataset_kwargs)
    test_dataset = CFSAilmentDataset(indices=test_indices, **dataset_kwargs) if test_size > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        if test_dataset is not None
        else None
    )

    stats = {
        "total": total,
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset) if test_dataset is not None else 0,
        "num_labels": train_dataset.num_labels,
    }

    print_info("Loaded CFS Visit 5 dataset from CSV")
    print_key_value("Total EDF files", stats["total"])
    print_key_value("Train samples", stats["train"])
    print_key_value("Validation samples", stats["val"])
    print_key_value("Test samples", stats["test"])
    print_key_value("Target sample rate", target_sample_rate)

    return train_loader, val_loader, test_loader, stats


