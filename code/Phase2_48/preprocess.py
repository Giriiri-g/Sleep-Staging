import os
import re
import glob
import torch
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

EPOCH_SEC = 30
TARGET_CHANNEL = "EEG Fpz-Cz"
PRE_POST_WAKE_MIN = 30  # minutes

OUTPUT_DIR = "processed_sleepedf"
TENSOR_DIR = os.path.join(OUTPUT_DIR, "tensors")
CSV_PATH = os.path.join(OUTPUT_DIR, "index.csv")

os.makedirs(TENSOR_DIR, exist_ok=True)

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

INVALID_STAGES = {"Sleep stage ?", "Movement time"}

# ============================================================
# Helpers
# ============================================================

def find_hypnogram(psg_file, directory):
    base = psg_file[:6]
    pattern = re.compile(rf"{base}..-Hypnogram.edf")
    for f in os.listdir(directory):
        if pattern.fullmatch(f):
            return os.path.join(directory, f)
    return None


def build_epoch_labels(raw, annotations):
    """
    Create per-epoch labels aligned with fixed 30s windows
    """
    sfreq = raw.info["sfreq"]
    total_samples = raw.n_times
    samples_per_epoch = int(EPOCH_SEC * sfreq)
    n_epochs = total_samples // samples_per_epoch

    # Initialize all epochs as invalid
    labels = np.array(["INVALID"] * n_epochs, dtype=object)

    for desc, onset, duration in zip(
        annotations.description,
        annotations.onset,
        annotations.duration
    ):
        if desc in INVALID_STAGES:
            continue
        if desc not in STAGE_MAP:
            continue

        start_epoch = int(onset // EPOCH_SEC)
        end_epoch = int((onset + duration) // EPOCH_SEC)

        labels[start_epoch:end_epoch] = STAGE_MAP[desc]

    return labels, samples_per_epoch


def extract_epochs_and_trim(raw, labels, samples_per_epoch):
    """
    Apply trimming in epoch space, then extract raw signal accordingly
    """
    # Remove invalid epochs
    valid_mask = labels != "INVALID"
    labels = labels[valid_mask]

    # Find sleep boundaries
    sleep_idx = np.where(labels != "W")[0]
    if len(sleep_idx) == 0:
        return None, None

    pre_epochs = int((PRE_POST_WAKE_MIN * 60) / EPOCH_SEC)

    start_ep = max(0, sleep_idx[0] - pre_epochs)
    end_ep = min(len(labels), sleep_idx[-1] + pre_epochs + 1)

    labels = labels[start_ep:end_ep]

    # Extract raw signal aligned with trimmed epochs
    start_sample = start_ep * samples_per_epoch
    end_sample = end_ep * samples_per_epoch

    data = raw.get_data(picks=[TARGET_CHANNEL])[0]
    data = data[start_sample:end_sample]

    # Reshape into epochs
    epochs = data.reshape(-1, samples_per_epoch)

    return epochs, labels.tolist()


# ============================================================
# Main processing
# ============================================================

def process_directories(directories):
    records = []

    for directory in directories:
        psg_files = glob.glob(os.path.join(directory, "*-PSG.edf"))

        for psg_path in tqdm(psg_files, desc=f"Processing {directory}"):
            fname = os.path.basename(psg_path)
            hyp_path = find_hypnogram(fname, directory)
            if hyp_path is None:
                continue

            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            if TARGET_CHANNEL not in raw.ch_names:
                continue

            annotations = mne.read_annotations(hyp_path)
            raw.set_annotations(annotations)

            labels, samples_per_epoch = build_epoch_labels(raw, annotations)
            epochs, labels = extract_epochs_and_trim(
                raw, labels, samples_per_epoch
            )

            if epochs is None or len(labels) == 0:
                continue

            tensor = torch.tensor(epochs, dtype=torch.float32)
            tensor_path = os.path.join(
                TENSOR_DIR, fname.replace(".edf", ".pt")
            )
            torch.save(tensor, tensor_path)

            records.append({
                "tensor_path": tensor_path,
                "stage_sequence": " ".join(labels)
            })

    return records


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    sleepedf_dirs = [
        r"sleep-edf-database-expanded-1.0.0\sleep-cassette",
        r"sleep-edf-database-expanded-1.0.0\sleep-telemetry"
    ]

    index = process_directories(sleepedf_dirs)
    df = pd.DataFrame(index)
    df.to_csv(CSV_PATH, index=False)

    print(f"Saved {len(df)} recordings")
    print(f"Tensors - {TENSOR_DIR}")
    print(f"Index CSV - {CSV_PATH}")
