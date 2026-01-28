import os
import re
import mne
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

PSG_DIR = r"sleep-edf-database-expanded-1.0.0\sleep-cassette"
PSG_FILE = "SC4001E0-PSG.edf"   # <-- change if needed

EPOCH_SEC = 30

STAGE_MAP = {
    'Sleep stage W': 'Wake',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N4',
    'Sleep stage R': 'REM'
}

STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'N4', 'REM']

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


def get_first_stage_epochs(annotations, sfreq):
    """
    Returns:
        dict(stage -> (start_sample, end_sample))
    """
    stage_epochs = {}

    for desc, onset, dur in zip(
        annotations.description,
        annotations.onset,
        annotations.duration
    ):
        if desc not in STAGE_MAP:
            continue

        stage = STAGE_MAP[desc]

        if stage in stage_epochs:
            continue

        if dur < EPOCH_SEC:
            continue

        start_sample = int(onset * sfreq)
        end_sample = start_sample + int(EPOCH_SEC * sfreq)

        stage_epochs[stage] = (start_sample, end_sample)

    return stage_epochs


# ============================================================
# Load PSG and annotations
# ============================================================

psg_path = os.path.join(PSG_DIR, PSG_FILE)
hyp_path = find_hypnogram(PSG_FILE, PSG_DIR)

raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
annotations = mne.read_annotations(hyp_path)
raw.set_annotations(annotations)

sfreq = raw.info['sfreq']
stage_epochs = get_first_stage_epochs(annotations, sfreq)

# ============================================================
# Plot per channel (ALL channels)
# ============================================================

for ch_idx, ch_name in enumerate(raw.ch_names):

    fig, axes = plt.subplots(
        len(STAGE_ORDER),
        1,
        figsize=(12, 9),
        sharex=True
    )
    fig.suptitle(f"{ch_name}")

    for ax, stage in zip(axes, STAGE_ORDER):
        if stage not in stage_epochs:
            ax.set_ylabel(stage)
            ax.text(0.5, 0.5, "Not present", ha='center', va='center')
            ax.set_yticks([])
            continue

        start, end = stage_epochs[stage]
        data = raw.get_data(picks=[ch_idx], start=start, stop=end)[0]

        time = np.arange(len(data)) / sfreq

        ax.plot(time, data, linewidth=0.8)
        ax.set_ylabel(stage)
        ax.set_yticks([])

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()
