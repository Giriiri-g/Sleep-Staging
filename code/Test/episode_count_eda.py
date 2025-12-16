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

def find_annotation_file_wildcard(psg_filename):
    base = psg_filename[:6]
    pattern = re.compile(rf'{base}..-Hypnogram.edf')
    for f in os.listdir(folder_path):
        if pattern.fullmatch(f):
            return os.path.join(folder_path, f)
    return None

# Dictionary to count number of episodes per stage across all files
episode_counts = defaultdict(int)

for psg_file in os.listdir(folder_path):
    if psg_file.endswith('-PSG.edf'):
        psg_path = os.path.join(folder_path, psg_file)
        hyp_path = find_annotation_file_wildcard(psg_file)
        if not hyp_path:
            print(f"Warning: No annotation file found for {psg_file}")
            continue

        annotations = mne.read_annotations(hyp_path)
        prev_stage = None

        # Iterate over annotation descriptions sequentially
        for desc in annotations.description:
            stage = stage_mapping.get(desc, 'Unscored')

            # Count a new episode only if current stage differs from previous
            if stage != prev_stage:
                episode_counts[stage] += 1  
                prev_stage = stage

print("Number of stage episodes in the dataset:")
for stage, count in episode_counts.items():
    print(f"{stage}: {count}")

stage, seconds = zip(*episode_counts.items())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
bars = plt.bar(stage, seconds, color='skyblue')

# Add labels and title
plt.xlabel('Sleep Stage')
plt.ylabel('Episode Count')
plt.title('Sleep Stage Episode Distribution')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()