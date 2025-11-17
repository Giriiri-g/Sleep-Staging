import mne
import glob
import os
from collections import defaultdict
import xml.etree.ElementTree as ET

path_signal = r"D:\cfs\polysomnography\edfs"
path_annotations_stages = r"D:\cfs\polysomnography\annotations-events-profusion"
path_annotations_events = r"D:\cfs\polysomnography\annotations-events-nsrr"

stage_mapping = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    'Stage 4 sleep|4': 4,
    "REM sleep|5": 5,
    'Unscored|9':6,
}


window_size = 30  # epoch length in seconds

full_segments = defaultdict(int)
leftover_seconds = defaultdict(float)


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    onsets = []
    durations = []
    descriptions = []
    #for event in root.iter('ScoredEvents'):
    # Loop through events in XML
    for event in root.findall('.//ScoredEvent'):
        start = float(event.find('Start').text) if event.find('Start') is not None else None # seconds
        duration = float(event.find('Duration').text) if event.find('Duration') is not None else None
        event_type = event.find('EventType').text if event.find('EventType') is not None else None
    # print("onset ",onset,"desc",description)
        # Only keep sleep stage events
        if event_type is not None and "Stages|Stages" in event_type:
            stage = event.find('EventConcept').text
            onsets.append(start)
            durations.append(duration)
            descriptions.append(stage)

    # Convert to MNE Annotations
    annotations = mne.Annotations(onset=onsets,
                                duration=durations,
                                description=descriptions)

    return annotations




for psg_file in os.listdir(path_signal):
    psg_path = os.path.join(path_signal, psg_file)
    annotations = parse_xml(r"D:\cfs\polysomnography\annotations-events-nsrr"+'\\' + psg_file[:17]+'-nsrr.xml')

    for desc, duration in zip(annotations.description, annotations.duration):
        stage = stage_mapping[desc]
        full_count = int(duration // window_size)
        leftover = duration % window_size
        full_segments[stage] += full_count
        leftover_seconds[stage] += leftover


# # edf = mne.io.read_raw_edf()

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
plt.title('MESA Sleep Stage segment Distribution')
plt.xticks([0, 1, 2, 3, 4, 5, 6], stage_mapping.keys())

# Display the exact seconds on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
