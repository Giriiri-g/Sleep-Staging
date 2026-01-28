import os
import re
import mne
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

BASE_DIRS = [
    r"sleep-edf-database-expanded-1.0.0\sleep-cassette",
    r"sleep-edf-database-expanded-1.0.0\sleep-telemetry"
]

EPOCH_SEC = 30
MAX_WAKE_SEC = 30 * 60  # 30 minutes

RAW_TO_STAGE = {
    'Sleep stage W': 'W',
    'Sleep stage R': 'R',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N3',   # merge N4 → N3
}

STAGE_COLORS = {
    'W': 'blue',
    'R': 'red',
    'N1': 'green',
    'N2': 'orange',
    'N3': 'purple'
}

# ============================================================
# Utilities
# ============================================================

def find_hypnogram(psg_file, directory):
    base = psg_file[:6]
    pattern = re.compile(rf"{base}..-Hypnogram.edf")
    for f in os.listdir(directory):
        if pattern.fullmatch(f):
            return os.path.join(directory, f)
    return None


def get_first_n_psg(n=5):
    files = []
    for directory in BASE_DIRS:
        for f in os.listdir(directory):
            if f.endswith("-PSG.edf"):
                files.append((directory, f))
                if len(files) == n:
                    return files
    return files


def load_segments(hyp_path):
    ann = mne.read_annotations(hyp_path)
    segments = []
    for s, o, d in zip(ann.description, ann.onset, ann.duration):
        if s in RAW_TO_STAGE:
            segments.append((RAW_TO_STAGE[s], o, d))
    return segments


# ============================================================
# Cleaning logic (episode-aware, time-based)
# ============================================================

def clean_segments(segments):
    if not segments:
        return []

    labels = [s for s, _, _ in segments]
    non_wake_idx = [i for i, s in enumerate(labels) if s != 'W']

    if not non_wake_idx:
        return segments

    first_sleep = non_wake_idx[0]
    last_sleep = non_wake_idx[-1]

    cleaned = []

    # ---- Wake BEFORE sleep (clip to 30 min) ----
    wake_time = 0
    for i in range(first_sleep):
        stage, onset, dur = segments[i]
        if stage != 'W':
            continue

        remaining = MAX_WAKE_SEC - wake_time
        if remaining <= 0:
            break

        clipped = min(dur, remaining)
        cleaned.append((stage, onset, clipped))
        wake_time += clipped

    # ---- Full sleep period ----
    for i in range(first_sleep, last_sleep + 1):
        cleaned.append(segments[i])

    # ---- Wake AFTER sleep (clip to 30 min) ----
    wake_time = 0
    for i in range(last_sleep + 1, len(segments)):
        stage, onset, dur = segments[i]
        if stage != 'W':
            continue

        remaining = MAX_WAKE_SEC - wake_time
        if remaining <= 0:
            break

        clipped = min(dur, remaining)
        cleaned.append((stage, onset, clipped))
        wake_time += clipped

    return cleaned


# ============================================================
# Plotting
# ============================================================

def plot_hypnograms(segments_list, title):
    fig, axes = plt.subplots(len(segments_list), 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title)

    for ax, (psg_name, segments) in zip(axes, segments_list):
        x = 0  # cumulative time in 30s units

        for stage, _, dur in segments:
            width = dur / EPOCH_SEC
            ax.barh(
                0,
                width,
                left=x,
                color=STAGE_COLORS[stage],
                edgecolor='black',
                height=0.8
            )
            x += width

        ax.set_title(psg_name, fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)

    axes[-1].set_xlabel("Time (30-second epochs)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS.values()]
    fig.legend(handles, STAGE_COLORS.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

psg_files = get_first_n_psg(5)

uncleaned = []
cleaned = []

for directory, psg in psg_files:
    hyp = find_hypnogram(psg, directory)
    if hyp is None:
        continue

    segments = load_segments(hyp)

    uncleaned.append((psg, segments))
    cleaned.append((psg, clean_segments(segments)))

plot_hypnograms(uncleaned, "Uncleaned Sleep Stage Sequences")
plot_hypnograms(cleaned, "Cleaned Sleep Stage Sequences")
