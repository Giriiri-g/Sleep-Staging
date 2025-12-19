import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

# --------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------

# Existing CSV you want to preserve
MESA_SELECTED_CSV = r"csv-docs/mesa_selected.csv"

# NSRR MESA phenotypes / PSG features
MESA_FEATURES_CSV = r"csv-docs/mesa-sleep-dataset-0.8.0.csv"

# Location of the MESA XML annotation files
XML_DIR = r"D:\mesa\annotations-events-nsrr"

# Final comprehensive output
OUTPUT_CSV = "mesa_comprehensive.csv"

# Sleep stage mapping from XML EventConcept codes
STAGE_MAP = {
    "0": 0,  # Wake
    "1": 1,  # N1
    "2": 2,  # N2
    "3": 3,  # N3
    "4": 4,  # REM
    "5": 5,  # Movement/Unknown
}


def safe_text(node):
    if node is None or node.text is None:
        return None
    return node.text.strip()


def parse_mesa_xml(xml_path: str):
    """Parse a single MESA XML file into a per‑epoch stage sequence."""
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

    # Remove initial wake if present (common scoring artifact)
    while stages_expanded and stages_expanded[0] == 0:
        stages_expanded.pop(0)

    return stages_expanded


def extract_mesa_id_from_path(path: str):
    """
    Extract numeric MESA ID from an EDF path like:
        .../mesa-sleep-1.edf  -> 1
        .../mesa-sleep-1013.edf -> 1013
    Returns None if it cannot be parsed (e.g., CFS EDFs).
    """
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"mesa-sleep-(\d+)", base)
    if not m:
        return None
    return int(m.group(1))


def build_sleep_stages_for_row(base_name: str):
    """
    Given an EDF base name (without extension), build its XML path and
    return the encoded sleep stages string, or '' if XML is missing.
    """
    xml_path = os.path.join(XML_DIR, base_name + "-nsrr.xml")
    if not os.path.exists(xml_path):
        # XML may be missing for some paths (e.g., non‑MESA studies)
        return ""

    try:
        stages = parse_mesa_xml(xml_path)
    except ET.ParseError:
        # Malformed XML – fail gracefully
        return ""

    return "".join(map(str, stages))


def main():
    # ----------------------------------------------------------------
    # 1. Load the existing disease label CSV (preserve as‑is)
    # ----------------------------------------------------------------
    df_sel = pd.read_csv(MESA_SELECTED_CSV)

    # Keep only MESA EDFs (paths that contain 'mesa'); drop CFS entries
    df_sel = df_sel[df_sel["path"].astype(str).str.contains("mesa", case=False)]
    df_sel = df_sel.reset_index(drop=True)

    # Keep original path column name
    if "path" not in df_sel.columns:
        raise ValueError(f"'path' column not found in {MESA_SELECTED_CSV}")

    # Base EDF name (e.g., mesa-sleep-1)
    df_sel["edf_base"] = df_sel["path"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )

    # Numeric MESA ID (None/NaN for non‑MESA EDFs like CFS)
    df_sel["mesaid"] = df_sel["path"].apply(extract_mesa_id_from_path)

    # ----------------------------------------------------------------
    # 2. Load MESA PSG feature dataset and subset requested columns
    # ----------------------------------------------------------------
    features = pd.read_csv(MESA_FEATURES_CSV)

    # Columns corresponding to the features you requested
    feature_cols = [
        # Apnea–Hypopnea Index (AASM-style AHI with 3% desat or arousal)
        "ahi_a0h3",
        # Arousal Index
        "ai_all5",
        # Oxygen Desaturation Index (3%)
        "odi35",
        # Percentages of sleep stages
        "timest1p5",   # % Stage 1
        "timest2p5",   # % Stage 2
        "times34p5",   # % Stage 3/4
        "timeremp5",   # % REM
        # Sleep efficiency, latency, WASO
        "slp_eff5",    # Sleep efficiency (%)
        "slp_lat5",    # Sleep latency (min)
        "waso5",       # Wake after sleep onset (min)
        # PLMI
        "plmaslp5",    # Periodic limb movements index during sleep
        # Total sleep time
        "slpprdp5",    # Total sleep time (min)
        # REM latency (sleep onset REM metric)
        "remlaiip5",
    ]

    # Ensure requested feature columns actually exist
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in MESA dataset: {missing}")

    features_subset = features[["mesaid"] + feature_cols]

    # ----------------------------------------------------------------
    # 3. Merge disease labels with PSG features on mesaid
    # ----------------------------------------------------------------
    df_merged = df_sel.merge(features_subset, on="mesaid", how="left")

    # ----------------------------------------------------------------
    # 4. Compute sleep stages strings from XML for each EDF
    # ----------------------------------------------------------------
    df_merged["sleep_stages"] = df_merged["edf_base"].apply(build_sleep_stages_for_row)

    # Left‑pad / right‑pad sleep stage sequences to a common length
    if not df_merged["sleep_stages"].empty:
        max_len = df_merged["sleep_stages"].map(len).max()
        df_merged["sleep_stages"] = df_merged["sleep_stages"].apply(
            lambda s: str(s).ljust(max_len, "0")
        )

    # ----------------------------------------------------------------
    # 5. Reorder/limit columns for a clean comprehensive CSV
    # ----------------------------------------------------------------
    # Start with original columns, then add sleep_stages and features
    base_cols = [
        "path",          # EDF path (preserved)
        "insomnia",
        "restless leg",
        "apnea",
        "snoring",
    ]

    # Only keep base columns that actually exist (in case labels change)
    base_cols = [c for c in base_cols if c in df_merged.columns]

    final_cols = (
        base_cols
        + ["sleep_stages"]
        + feature_cols
    )

    df_out = df_merged[final_cols]

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Created comprehensive CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
