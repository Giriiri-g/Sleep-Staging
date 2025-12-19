import os
import csv
import glob
import xml.etree.ElementTree as ET
import pandas as pd

XML_DIR = r"D:\mesa\annotations-events-nsrr"
OUTPUT_CSV = "sleep_stages_output.csv"

STAGE_MAP = {
    "0": 0,  # Wake
    "1": 1,  # N1
    "2": 2,  # N2
    "3": 3,  # N3
    "4": 4,  # REM
    "5": 5   # Movement/Unknown
}

def safe_text(node):
    if node is None or node.text is None:
        return None
    return node.text.strip()

def parse_mesa_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stages_expanded = []

    for ev in root.iter("ScoredEvent"):

        event_type = safe_text(ev.find("EventType"))
        concept = safe_text(ev.find("EventConcept"))
        duration_text = safe_text(ev.find("Duration"))

        if event_type is None or concept is None or duration_text is None:
            continue

        if not event_type.startswith("Stages|"):
            continue

        duration = float(duration_text)
        stage_code = concept.split("|")[-1].strip()

        if stage_code not in STAGE_MAP:
            continue

        stage_val = STAGE_MAP[stage_code]

        repeats = int(duration // 30)
        stages_expanded.extend([stage_val] * repeats)

    # ----------------------------------------------------------
    # NEW PART: Remove initial wake episode if it exists
    # ----------------------------------------------------------
    while len(stages_expanded) > 0 and stages_expanded[0] == 0:
        stages_expanded.pop(0)

    return stages_expanded


def main():
    xml_files = sorted(glob.glob(os.path.join(XML_DIR, "*.xml")))
    rows = []

    for xml_path in xml_files:
        stages = parse_mesa_xml(xml_path)
        rows.append([xml_path, ''.join(map(str, stages))])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mesaid", "sleep_stages"])
        writer.writerows(rows)
        print(rows[0])

    print("Created:", OUTPUT_CSV)
    df = pd.read_csv(OUTPUT_CSV)
    MAX_LEN = df["sleep_stages"].map(len).max()
    df['sleep_stages'] = df['sleep_stages'].apply(lambda x: x.ljust(MAX_LEN, '0'))
    df['mesaid'] = df['mesaid'].apply(lambda x: x.split('-')[-2])
    df.to_csv(OUTPUT_CSV, index=False)
if __name__ == "__main__":
    main()
