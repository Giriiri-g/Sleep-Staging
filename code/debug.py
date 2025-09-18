import os
import mne
from collections import Counter, defaultdict

# Path to your hypnogram EDF folder
HYPNO_PATH = r"sleep-edf-database-expanded-1.0.0\sleep-cassette"

# Mapping of label strings to canonical classes
stage_map = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage N4": "N3",  # sometimes present in older scoring
    "Sleep stage R": "REM",
    "Sleep stage ?": "?",
    "Movement time": "M"
}

# Store global distribution
global_counts = Counter()
file_counts = defaultdict(Counter)

for root, _, files in os.walk(HYPNO_PATH):
    for fname in files:
        if fname.lower().endswith(".edf"):
            fpath = os.path.join(root, fname)
            try:
                annots = mne.read_annotations(fpath)
                stages = [stage_map.get(desc, desc) for desc in annots.description]
                cnt = Counter(stages)
                file_counts[fname] = cnt
                global_counts.update(cnt)
            except Exception as e:
                print(f"Could not read {fname}: {e}")

# Print per-file counts
for fname, cnt in file_counts.items():
    print(f"\n{fname}:")
    for stage, num in cnt.items():
        print(f"  {stage}: {num}")

# Print overall distribution
print("\n=== Global Distribution ===")
for stage, num in global_counts.items():
    print(f"{stage}: {num}")
