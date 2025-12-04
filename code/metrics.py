import mne
import xml.etree.ElementTree as ET

def extract_sleep_stages_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    onsets = []
    durations = []
    descriptions = []

    for event in root.findall('.//ScoredEvent'):
        start = float(event.find('Start').text) if event.find('Start') is not None else None  # seconds
        duration = float(event.find('Duration').text) if event.find('Duration') is not None else None
        event_type = event.find('EventType').text if event.find('EventType') is not None else None

        if event_type is not None and "Stages|Stages" in event_type:
            stage = event.find('EventConcept').text
            onsets.append(start)
            durations.append(duration)
            descriptions.append(stage)

    annotations = mne.Annotations(onset=onsets,
                                  duration=durations,
                                  description=descriptions)

    # Optionally, return a list of tuples for sleep stages
    sleep_stages_list = list(zip(onsets, durations, descriptions))
    
    return annotations, sleep_stages_list


# Example usage
xml_path = r"C:\Users\haris\Downloads\shhs1-200002-nsrr.xml"
annotations, stages_list = extract_sleep_stages_from_xml(xml_path)

print("Extracted sleep stages:")
for onset, duration, desc in stages_list:
    print(f"Onset: {onset:.2f}s, Duration: {duration:.2f}s, Stage: {desc}")
