import os
import re
import mne
from collections import defaultdict

folder_path = r'C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette'

stage_mapping = {
    'Sleep stage W': 'Wake',
    'Sleep stage R': 'REM',
    'Sleep stage 1': 'Stage 1',
    'Sleep stage 2': 'Stage 2',
    'Sleep stage 3': 'Stage 3',
    'Sleep stage 4': 'Stage 4',
    'Sleep stage ?': 'Unscored'
}

class_seconds = defaultdict(float)

def find_annotation_file_wildcard(psg_filename):
    base = psg_filename[:6]
    pattern = re.compile(rf'{base}..-Hypnogram.edf')
    for f in os.listdir(folder_path):
        if pattern.fullmatch(f):
            return os.path.join(folder_path, f)
    return None

for psg_file in os.listdir(folder_path):
    if psg_file.endswith('-PSG.edf'):
        psg_path = os.path.join(folder_path, psg_file)
        hyp_path = find_annotation_file_wildcard(psg_file)
        if not hyp_path:
            print(f"Warning: No annotation file found for {psg_file}")
            continue

        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations)

        for desc, dur in zip(annotations.description, annotations.duration):
            stage = stage_mapping.get(desc, 'Unscored')
            class_seconds[stage] += dur//30

print("Class distribution (epochs):")
for stage, seconds in class_seconds.items():
    print(f"{stage}: {seconds:.2f}")
stage, seconds = zip(*class_seconds.items())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
bars = plt.bar(stage, seconds, color='skyblue')

# Add labels and title
plt.xlabel('Sleep Stage')
plt.ylabel('Duration (epochs)')
plt.title('Sleep Stage segments Distribution')

# Display the exact seconds on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
