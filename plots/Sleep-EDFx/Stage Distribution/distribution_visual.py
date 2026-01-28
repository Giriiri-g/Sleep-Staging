import os
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
FILES = [
    r"sleep-edf-database-expanded-1.0.0\sleep-cassette",
    r"sleep-edf-database-expanded-1.0.0\sleep-telemetry"
]

EPOCH_SEC = 30
wake_clipping = 0
STAGE_MAP = {
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N4",
    "Sleep stage R": "REM",
    "Sleep stage W": "Wake",
    "Sleep stage ?": "Unscored",
    "Movement time": "Movement"
}

# --------------------------------------------------
# Containers
# --------------------------------------------------
epoch_count = {}
episode_count = {}

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for directory in FILES:
    hyp_files = glob.glob(os.path.join(directory, "*-Hypnogram.edf"))
    print(f"Found {len(hyp_files)} hypnogram files in {directory}")

    for file in hyp_files:
        annot = mne.read_annotations(file)

        prev_stage = None

        for desc, dur in zip(annot.description, annot.duration):
            stage = STAGE_MAP.get(desc, None)
            if stage is None:
                continue

            # ---------- Episode count ----------
            if stage != prev_stage:
                episode_count[stage] = episode_count.get(stage, 0) + 1
                prev_stage = stage

            # ---------- Epoch count ----------
            epochs = int(round(dur / EPOCH_SEC))
            if stage == "Wake" and epochs >60:
                wake_clipping+= epochs - 60
            epoch_count[stage] = epoch_count.get(stage, 0) + epochs


# --------------------------------------------------
# Plot helper
# --------------------------------------------------
def plot_bar(data, title, ylabel):
    stages = list(data.keys())
    values = list(data.values())

    bars = plt.bar(stages, values)
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Sleep Stage")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# --------------------------------------------------
# UNCLLEANED PLOTS
# --------------------------------------------------
print("\nEpoch distribution (uncleaned):")
for k, v in epoch_count.items():
    print(f"{k}: {v}")

plot_bar(
    epoch_count,
    "Sleep Stage Epoch Distribution – Sleep-EDFx (Uncleaned)",
    "Epoch Count (30s)"
)

print("\nEpisode distribution (uncleaned):")
for k, v in episode_count.items():
    print(f"{k}: {v}")

plot_bar(
    episode_count,
    "Sleep Stage Episode Distribution – Sleep-EDFx (Uncleaned)",
    "Episode Count"
)

# --------------------------------------------------
# CLEANED DISTRIBUTIONS
# --------------------------------------------------
def clean_distribution(dist):
    dist = dist.copy()

    # Merge N4 → N3
    if "N4" in dist:
        dist["N3"] = dist.get("N3", 0) + dist["N4"]
        del dist["N4"]

    # Remove non-AASM
    dist.pop("Unscored", None)
    dist.pop("Movement", None)
    dist["Wake"] = dist.get("Wake", 0) - wake_clipping
    return dist


epoch_clean = clean_distribution(epoch_count)
episode_clean = clean_distribution(episode_count)

plot_bar(
    epoch_clean,
    "Sleep Stage Epoch Distribution – Sleep-EDFx (Cleaned)",
    "Epoch Count (30s)"
)

plot_bar(
    episode_clean,
    "Sleep Stage Episode Distribution – Sleep-EDFx (Cleaned)",
    "Episode Count"
)
