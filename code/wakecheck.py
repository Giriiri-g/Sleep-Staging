import os
import xml.etree.ElementTree as ET

def first_stage_not_wake(xml_path):
    """
    Returns True if the first Stages ScoredEvent is NOT Wake.
    Otherwise returns False.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find all ScoredEvent tags
    events = root.findall(".//ScoredEvent")

    for ev in events:
        event_type = ev.findtext("EventType")

        # Only consider sleep stages
        if event_type and "Stages" in event_type:
            event_concept = ev.findtext("EventConcept", "")

            # Extract stage name before "|" if present
            stage_name = event_concept.split("|")[0].strip()

            return stage_name != "Wake"

    # If no stage events found, treat as error
    return False


# -------- MAIN --------
xml_folder = r"D:\cfs\polysomnography\annotations-events-nsrr"

bad_files = []

for file in os.listdir(xml_folder):
    if file.lower().endswith(".xml"):
        fpath = os.path.join(xml_folder, file)
        if first_stage_not_wake(fpath):
            bad_files.append(file)

print("XML files where FIRST stage is NOT Wake:")
for bf in bad_files:
    print(" -", bf)

print("\nTotal:", len(bad_files))

