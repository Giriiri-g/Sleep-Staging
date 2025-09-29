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

window_size = 30  # epoch length in seconds

full_segments = defaultdict(int)
leftover_seconds = defaultdict(float)

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
            print(f"Warning: No annotation file found for {psg_file}, skipping.")
            continue

        annotations = mne.read_annotations(hyp_path)

        for desc, duration in zip(annotations.description, annotations.duration):
            stage = stage_mapping.get(desc, 'Unscored')
            full_count = int(duration // window_size)
            leftover = duration % window_size
            full_segments[stage] += full_count
            leftover_seconds[stage] += leftover

print("Aggregated sleep stage segmentation report across all records:")
for stage in full_segments:
    print(f"{stage}: Full 30s segments = {full_segments[stage]}, Leftover seconds = {leftover_seconds[stage]:.2f}")


import matplotlib.pyplot as plt

stages, seconds = zip(*full_segments.items())

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(stages, seconds, color='skyblue')

# Add labels and title
plt.xlabel('Sleep Stage')
plt.ylabel('Segment Count (30s each)')
plt.title('Sleep Stage segment Distribution')

# Display the exact seconds on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
