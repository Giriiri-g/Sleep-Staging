"""
MESA dataset preprocessing for sleep staging.

Uses both PSG (EDF) files and XML annotation files:
- EDF: Raw polysomnography signals (EEG, EOG, EMG, etc.)
- XML: NSRR-format sleep stage annotations (ScoredEvent elements with EventType=Stages|Stages)

Output format matches Sleep-EDF preprocess: 30s epochs, tensors + index.csv with stage_sequence.
"""

import os
import re
import glob
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# Configuration
# ============================================================

EPOCH_SEC = 30
# Single EEG channel: EEG1 (Fz-Cz) only
TARGET_CHANNEL = "EEG1"
PRE_POST_WAKE_MIN = 30  # minutes of wake before/after sleep to retain

VALID_STAGES = {"W", "N1", "N2", "N3", "REM"}

# MESA input directories (standard NSRR layout)
MESA_ROOT = r"E:\mesa"
EDF_DIR = os.path.join(MESA_ROOT, "polysomnography", "edfs")
XML_DIR = os.path.join(MESA_ROOT, "polysomnography", "annotations-events-nsrr")

# Output
OUTPUT_DIR = r"E:\mesa\mesa_preprocessed"
TENSOR_DIR = os.path.join(OUTPUT_DIR, "tensors")
CSV_PATH = os.path.join(OUTPUT_DIR, "index.csv")

os.makedirs(TENSOR_DIR, exist_ok=True)

# MESA NSRR XML EventConcept -> standard stage labels (matching Sleep-EDF)
# EventConcept format: "Wake|0", "Stage 1 sleep|1", "Stage 2 sleep|2", etc.
STAGE_MAP = {
    "Wake|0": "W",
    "Stage 1 sleep|1": "N1",
    "Stage 2 sleep|2": "N2",
    "Stage 3 sleep|3": "N3",
    "Stage 4 sleep|4": "N3",
    "REM sleep|5": "REM",
}
# Alternative EventConcept formats (some MESA files may differ)
STAGE_MAP_ALT = {
    "0": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "5": "REM",
}
# ============================================================
# XML parsing
# ============================================================


def parse_mesa_xml_labels(xml_path: str) -> list[tuple[float, float, str]]:
    """
    Parse MESA NSRR XML and return list of (start_sec, duration_sec, stage_label) for sleep stages.

    ScoredEvent elements with EventType starting with "Stages|" contain sleep stages.
    EventConcept values: "Wake|0", "Stage 1 sleep|1", etc.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []

    def safe_text(node):
        if node is None or node.text is None:
            return None
        return node.text.strip()

    for ev in root.iter("ScoredEvent"):
        event_type = safe_text(ev.find("EventType"))
        concept = safe_text(ev.find("EventConcept"))
        start_el = ev.find("Start")
        duration_el = ev.find("Duration")

        if event_type is None or concept is None or start_el is None or duration_el is None:
            continue
        if not event_type.startswith("Stages|"):
            continue

        try:
            start_sec = float(start_el.text)
            duration_sec = float(duration_el.text)
        except (TypeError, ValueError):
            continue

        stage_label = None
        if concept in STAGE_MAP:
            stage_label = STAGE_MAP[concept]
        else:
            # Try extracting stage code from concept (e.g. "Wake|0" -> "0")
            code = concept.split("|")[-1].strip() if "|" in concept else concept
            if code in STAGE_MAP_ALT:
                stage_label = STAGE_MAP_ALT[code]

        if stage_label is not None:
            events.append((start_sec, duration_sec, stage_label))

    return events


def build_epoch_labels_from_xml(
    xml_path: str,
    total_duration_sec: float,
) -> tuple[np.ndarray, int]:
    """
    Build per-epoch labels from XML events, aligned with fixed 30s windows.

    Returns:
        labels: array of stage strings (W, N1, N2, N3, REM) for each 30s epoch
        samples_per_epoch: number of samples per 30s epoch (depends on sfreq)
    """
    events = parse_mesa_xml_labels(xml_path)
    n_epochs = int(total_duration_sec // EPOCH_SEC)
    labels = np.array(["INVALID"] * n_epochs, dtype=object)

    for start_sec, duration_sec, stage_label in events:
        start_epoch = int(start_sec // EPOCH_SEC)
        end_epoch = int((start_sec + duration_sec) // EPOCH_SEC)
        end_epoch = min(end_epoch, n_epochs)
        if start_epoch >= n_epochs:
            continue
        labels[start_epoch:end_epoch] = stage_label

    return labels, n_epochs


# ============================================================
# File discovery
# ============================================================


def find_xml_for_edf(edf_path: str, xml_dir: str) -> str | None:
    """
    Find corresponding NSRR XML for a MESA EDF file.
    EDF: mesa-sleep-{id}.edf
    XML: mesa-sleep-{id}-nsrr.xml  OR  {id}-nsrr.xml (in annotations-events-nsrr)
    """
    base = os.path.splitext(os.path.basename(edf_path))[0]
    xml_candidates = [
        os.path.join(xml_dir, f"{base}-nsrr.xml"),
    ]
    m = re.search(r"mesa-sleep-(\d+)", base, re.IGNORECASE)
    if m:
        nsrrid = m.group(1)
        xml_candidates.append(os.path.join(xml_dir, f"{nsrrid}-nsrr.xml"))
        xml_candidates.append(os.path.join(xml_dir, f"mesa-sleep-{nsrrid}-nsrr.xml"))

    for p in xml_candidates:
        if os.path.exists(p):
            return p
    return None


def get_target_channel(raw: mne.io.Raw) -> str | None:
    """Return EEG1 (Fz-Cz) if available, else None."""
    return TARGET_CHANNEL if TARGET_CHANNEL in raw.ch_names else None


# ============================================================
# Epoch extraction and trimming
# ============================================================


def extract_epochs_and_trim(
    raw: mne.io.Raw,
    labels: np.ndarray,
    samples_per_epoch: int,
    target_channel: str,
) -> tuple[np.ndarray | None, list | None]:
    """
    Remove invalid epochs, trim to sleep period + pre/post wake, extract signal.
    Only W, N1, N2, N3, REM are produced (STAGE_MAP maps unknown stages to None/not added).
    """
    valid_mask = labels != "INVALID"
    labels = labels[valid_mask]

    sleep_idx = np.where(labels != "W")[0]
    if len(sleep_idx) == 0:
        return None, None

    pre_epochs = int((PRE_POST_WAKE_MIN * 60) / EPOCH_SEC)
    start_ep = max(0, sleep_idx[0] - pre_epochs)
    end_ep = min(len(labels), sleep_idx[-1] + pre_epochs + 1)

    labels = labels[start_ep:end_ep]

    start_sample = start_ep * samples_per_epoch
    end_sample = end_ep * samples_per_epoch

    data = raw.get_data(picks=[target_channel])[0]
    data = data[start_sample:end_sample]

    epochs = data.reshape(-1, samples_per_epoch)
    return epochs, labels.tolist()


# ============================================================
# Main processing
# ============================================================


def process_mesa(
    edf_dir: str,
    xml_dir: str,
    output_dir: str,
    limit: int | None = None,
) -> list[dict]:
    """
    Process MESA EDF + XML pairs. Writes tensors to output_dir/tensors and returns records.
    """
    tensor_dir = os.path.join(output_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    records = []
    edf_files = glob.glob(os.path.join(edf_dir, "*.edf"))
    total = len(edf_files)
    if limit is not None:
        edf_files = edf_files[:limit]
        print(f"Processing first {len(edf_files)} of {total} EDF files")

    for idx, psg_path in enumerate(tqdm(edf_files, desc="Processing MESA"), start=1):
        fname = os.path.basename(psg_path)
        out_name = fname.replace(".edf", ".pt")
        tensor_path = os.path.join(tensor_dir, out_name)
        if os.path.exists(tensor_path):
            tqdm.write(f"[{idx}/{len(edf_files)}] {fname} already preprocessed, skipping")
            continue

        tqdm.write(f"[{idx}/{len(edf_files)}] {fname} (loading EDF...)")
        xml_path = find_xml_for_edf(psg_path, xml_dir)
        if xml_path is None:
            tqdm.write(f"  No XML for {fname}, skipping")
            continue

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        except Exception as e:
            tqdm.write(f"  Failed to load {fname}: {e}")
            continue

        target_ch = get_target_channel(raw)
        if target_ch is None:
            tqdm.write(f"  EEG1 (Fz-Cz) not found in {fname}, skipping")
            continue

        sfreq = raw.info["sfreq"]
        total_samples = raw.n_times
        total_duration_sec = total_samples / sfreq
        samples_per_epoch = int(EPOCH_SEC * sfreq)

        try:
            labels, _ = build_epoch_labels_from_xml(xml_path, total_duration_sec)
        except ET.ParseError as e:
            tqdm.write(f"  XML parse error for {fname}: {e}")
            continue

        epochs, stage_list = extract_epochs_and_trim(
            raw, labels, samples_per_epoch, target_ch
        )

        if epochs is None or len(stage_list) == 0:
            tqdm.write(f"  No valid sleep epochs for {fname}, skipping")
            continue

        tensor = torch.tensor(epochs, dtype=torch.float32)
        torch.save(tensor, tensor_path)

        records.append({
            "tensor_path": tensor_path,
            "stage_sequence": " ".join(stage_list),
        })

    return records


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MESA sleep dataset (PSG + XML)")
    parser.add_argument(
        "--mesa_root",
        type=str,
        default=MESA_ROOT,
        help="Root directory of MESA dataset (default: E:\\mesa)",
    )
    parser.add_argument(
        "--edf_dir",
        type=str,
        default=None,
        help="Override: directory containing MESA EDF files",
    )
    parser.add_argument(
        "--xml_dir",
        type=str,
        default=None,
        help="Override: directory containing MESA NSRR XML annotation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for tensors and index.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N EDF files (for testing)",
    )
    args = parser.parse_args()

    mesa_root = args.mesa_root
    edf_dir = args.edf_dir or os.path.join(mesa_root, "polysomnography", "edfs")
    xml_dir = args.xml_dir
    if xml_dir is None:
        # Try standard NSRR layout first, then alternative (xmls/mesa/...)
        std_xml = os.path.join(mesa_root, "polysomnography", "annotations-events-nsrr")
        alt_xml = os.path.join(mesa_root, "xmls", "mesa", "polysomnography", "annotations-events-nsrr")
        xml_dir = std_xml if os.path.isdir(std_xml) else alt_xml
    output_dir = args.output_dir

    if not os.path.isdir(edf_dir):
        print(f"ERROR: EDF directory not found: {edf_dir}")
        exit(1)
    if not os.path.isdir(xml_dir):
        print(f"ERROR: XML directory not found: {xml_dir}")
        exit(1)

    index = process_mesa(
        edf_dir=edf_dir,
        xml_dir=xml_dir,
        output_dir=output_dir,
        limit=args.limit,
    )
    df = pd.DataFrame(index)
    csv_path = os.path.join(output_dir, "index.csv")
    df.to_csv(csv_path, index=False)

    tensor_dir = os.path.join(output_dir, "tensors")
    print(f"Saved {len(df)} recordings")
    print(f"Tensors: {tensor_dir}")
    print(f"Index CSV: {csv_path}")
