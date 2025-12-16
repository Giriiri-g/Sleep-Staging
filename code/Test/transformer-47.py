"""
transformer_sleep_staging_multi.py

Lightweight Transformer for Sleep Staging (7-class) using all 7 PSG channels.
Each input epoch: (7 channels × 3000 samples) representing 30 s @100Hz.
Includes self-attention mechanism and multi-channel input support.
"""

import os
import glob
import argparse
import math
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import mne
    _EDF_READER = 'mne'
except Exception:
    _EDF_READER = None


# ---------------------------
# Preprocessing utils
# ---------------------------
def bandpass_filter(sig, sf, low=0.3, high=35.0):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [low, high], btype='bandpass', fs=sf, output='sos')
    return sosfiltfilt(sos, sig)

def resample_signal(sig, orig_fs, target_fs):
    from scipy.signal import resample
    n = int(round(len(sig) * float(target_fs) / orig_fs))
    return resample(sig, n)

def epoch_signal(sig, epoch_samples):
    n_epochs = sig.shape[1] // epoch_samples
    if n_epochs == 0:
        return np.zeros((0, sig.shape[0], epoch_samples))
    truncated = sig[:, :n_epochs * epoch_samples]
    return truncated.reshape(sig.shape[0], n_epochs, epoch_samples).transpose(1, 0, 2)  # (epochs, channels, samples)


# ---------------------------
# EDF loader
# ---------------------------
def load_edf_file(path, target_fs=100, epoch_seconds=30):
    """
    Load EDF with all 7 PSG channels, resample, filter, and epoch.
    Returns shape: (n_epochs, 7, epoch_samples)
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    orig_fs = raw.info['sfreq']

    # try to pick the 7 main PSG channels (EEG, EOG, EMG, ECG, etc.)
    wanted_channels = [
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental',
        'Temp rectal', 'Resp oro-nasal', 'Event marker'
    ]
    pick = [ch for ch in wanted_channels if ch in raw.ch_names]
    if len(pick) < 7:
        print(f"Warning: {path} has only {len(pick)} of 7 channels; using available ones.")
    raw.pick(pick)
    sig = raw.get_data()  # shape (channels, samples)

    # filter and resample each channel
    filtered = []
    for c in range(sig.shape[0]):
        x = bandpass_filter(sig[c], orig_fs)
        x = resample_signal(x, orig_fs, target_fs)
        filtered.append(x)
    sig = np.stack(filtered)
    epoch_samples = int(epoch_seconds * target_fs)
    epochs = epoch_signal(sig, epoch_samples)
    return epochs  # (epochs, channels, samples)


# ---------------------------
# Dataset
# ---------------------------
class SleepEDFFolderDataset(Dataset):
    def __init__(self, edf_folder, target_fs=100, epoch_seconds=30):
        self.edf_folder = edf_folder
        self.target_fs = target_fs
        self.epoch_seconds = epoch_seconds
        self.epoch_samples = int(target_fs * epoch_seconds)
        self.items = []
        self.cache = {}
        self._index_folder()

    def _index_folder(self):
        pattern = os.path.join(self.edf_folder, "*.edf")
        files = sorted(glob.glob(pattern))
        for f in files:
            try:
                epochs = load_edf_file(f, self.target_fs, self.epoch_seconds)
                for i in range(len(epochs)):
                    self.items.append((f, i))
                self.cache[f] = epochs
            except Exception as e:
                print(f"Skipping {f}: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fpath, eidx = self.items[idx]
        epochs = self.cache[fpath]
        sig = epochs[eidx].astype(np.float32)
        # normalize each channel independently
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
        return torch.from_numpy(sig), torch.tensor(-1, dtype=torch.long)


# ---------------------------
# Model
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class LightweightTransformer(nn.Module):
    def __init__(self, num_channels=7, input_samples=3000, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, num_classes=7, dropout=0.1):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        reduced_len = input_samples // 8
        self.pos_enc = PositionalEncoding(d_model, max_len=reduced_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (B, 7, 3000)
        x = self.frontend(x)  # (B, d_model, Lr)
        x = x.permute(0, 2, 1)  # (B, Lr, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        out = self.classifier(x)
        return out


# ---------------------------
# Train/eval
# ---------------------------
def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch])
    return xs, ys

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        mask = (y >= 0)
        correct += (preds[mask] == y[mask]).sum().item()
        total += mask.sum().item()
    return total_loss / len(loader.dataset), correct / total if total > 0 else 0

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            mask = (y >= 0)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    return total_loss / len(loader.dataset), correct / total if total > 0 else 0


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=str, default="checkpoint_7ch.pth")
    return p.parse_args()

def main():
    args = parse_args()
    print("Using device:", args.device)
    ds = SleepEDFFolderDataset(args.data_dir)
    print(f"Loaded {len(ds)} epochs from EDFs.")
    from torch.utils.data import random_split
    n = len(ds)
    n_train = int(0.8 * n)
    train_ds, val_ds = random_split(ds, [n_train, n - n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LightweightTransformer(num_channels=7, num_classes=7)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, args.device)
        print(f"Train loss {tr_loss:.4f} acc {tr_acc:.4f}")
        print(f"Val   loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.save)
            print(f"Saved best model -> {args.save}")

if __name__ == "__main__":
    main()
