import os
import torch
import numpy as np
import pandas as pd
import pywt
from scipy.signal import stft
from tqdm import tqdm

# =========================
# CONFIG (EDIT THESE)
# =========================
FS = 100
WAVELET = "db4"
DWT_LEVEL = 5

INPUT_CSV = r"C:\PS\Sleep-Staging\processed_sleepedf\index.csv"
OUTPUT_FEATURE_DIR = r"C:\PS\Sleep-Staging\processed_sleepedf\spectral"
OUTPUT_CSV = INPUT_CSV

os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)


# =========================
# Utility Functions
# =========================

def spectral_entropy(power_spectrum):
    ps = power_spectrum / (np.sum(power_spectrum) + 1e-12)
    return -np.sum(ps * np.log(ps + 1e-12))


def spectral_edge_frequency(freqs, power, edge=0.95):
    cumulative = np.cumsum(power)
    threshold = edge * cumulative[-1]
    idx = np.where(cumulative >= threshold)[0][0]
    return freqs[idx]


# =========================
# DWT Features
# =========================

def extract_dwt_features(epoch):
    coeffs = pywt.wavedec(epoch, WAVELET, level=DWT_LEVEL)
    features = []

    for c in coeffs:
        energy = np.sum(c ** 2)
        log_energy = np.log(energy + 1e-12)
        variance = np.var(c)

        p = (c ** 2) / (np.sum(c ** 2) + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))

        features.extend([energy, log_energy, entropy, variance])

    return features


# =========================
# STFT Features
# =========================

def extract_stft_features(epoch):
    f, t, Zxx = stft(epoch,
                     fs=FS,
                     window='hann',
                     nperseg=2*FS,
                     noverlap=FS)

    power = np.abs(Zxx) ** 2
    power_mean = np.mean(power, axis=1)
    total_power = np.sum(power_mean)

    def band_power(low, high):
        idx = np.logical_and(f >= low, f <= high)
        return np.sum(power_mean[idx])

    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta  = band_power(13, 30)
    gamma = band_power(30, 45)

    relative_powers = [
        delta / total_power,
        theta / total_power,
        alpha / total_power,
        beta / total_power,
        gamma / total_power
    ]

    entropy = spectral_entropy(power_mean)
    sef95 = spectral_edge_frequency(f, power_mean, edge=0.95)

    band_dict = {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta
    }

    return relative_powers + [entropy, sef95], band_dict


# =========================
# Ratio Features
# =========================

def extract_ratios(bands):
    delta = bands["delta"]
    theta = bands["theta"]
    alpha = bands["alpha"]
    beta = bands["beta"]

    return [
        delta / (beta + 1e-12),
        theta / (alpha + 1e-12),
        (delta + theta) / (alpha + beta + 1e-12)
    ]


# =========================
# Feature Extraction per File
# =========================

def process_file(file_path):

    data = torch.load(file_path)

    if isinstance(data, dict):
        eeg = data["eeg"]
    else:
        eeg = data

    eeg = eeg.numpy()

    if len(eeg.shape) == 1:
        eeg_epochs = eeg.reshape(1, -1)
    else:
        eeg_epochs = eeg

    all_features = []

    for epoch in eeg_epochs:
        features = []

        # DWT
        features.extend(extract_dwt_features(epoch))

        # STFT
        stft_feats, band_dict = extract_stft_features(epoch)
        features.extend(stft_feats)

        # Ratios
        features.extend(extract_ratios(band_dict))

        all_features.append(features)

    return torch.tensor(all_features, dtype=torch.float32)


# =========================
# Main
# =========================

def main():

    df = pd.read_csv(INPUT_CSV)

    if "tensor_path" not in df.columns:
        raise ValueError("CSV must contain a column named 'path'")

    spectral_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        input_path = row["tensor_path"]

        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            spectral_paths.append(None)
            continue

        features = process_file(input_path)

        base_name = os.path.basename(input_path).replace(".pt", "_spectral.pt")
        output_path = os.path.join(OUTPUT_FEATURE_DIR, base_name)

        torch.save(features, output_path)

        spectral_paths.append(output_path)

    df["spectral"] = spectral_paths
    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ Spectral extraction complete.")
    print(f"Updated CSV saved at: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
