"""
Offline preprocessing for CFS EDF signals.

Steps:
- Load EDF using MNE
- Find corresponding XML hypnogram
- Clip the first full wake episode (first sleep stage segment)
- Resample, select channels, normalize
- Interpolate/compress to fixed input_length
- Save as .pt tensor and rewrite CSV 'path' column to point to .pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET

from utils import print_info, print_warning, print_success, print_error


def find_annotation_file(edf_path: Path, annotation_dir: Path) -> Optional[Path]:
    """
    Find corresponding XML annotation file for a CFS EDF file.

    Mirrors logic in sleep_staging_transformer._find_annotation_file:
    - EDF:  cfs-visit5-{nsrrid}.edf
    - XML:  {nsrrid}-nsrr.xml
    """
    base_name = edf_path.stem
    # Primary pattern
    if "cfs-visit5-" in base_name:
        nsrrid = base_name.replace("cfs-visit5-", "")
        xml_filename = f"{nsrrid}-nsrr.xml"
        xml_path = annotation_dir / xml_filename
        if xml_path.exists():
            return xml_path

    # Fallback: any XML whose name contains the base name prefix
    for f in annotation_dir.glob("*.xml"):
        if base_name[:17] in f.name:
            return f

    return None


def parse_first_stage_from_xml(xml_path: Path) -> Optional[Tuple[float, float, str]]:
    """
    Parse XML hypnogram and return (start, duration, stage) of the first sleep stage event.

    We assume CFS XML files store events in <ScoredEvent> with:
    - <EventType>Stages|Stages</EventType>
    - <EventConcept>Stage string</EventConcept>
    - <Start>, <Duration> in seconds
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for event in root.findall(".//ScoredEvent"):
            start_el = event.find("Start")
            duration_el = event.find("Duration")
            type_el = event.find("EventType")
            concept_el = event.find("EventConcept")

            if (
                start_el is None
                or duration_el is None
                or type_el is None
                or concept_el is None
            ):
                continue

            event_type = type_el.text or ""
            if "Stages|Stages" not in event_type:
                continue

            start = float(start_el.text)
            duration = float(duration_el.text)
            stage = concept_el.text or ""

            return start, duration, stage

        return None
    except Exception as e:
        print_warning(f"Failed to parse XML {xml_path}: {e}")
        return None


def extract_and_preprocess_signal(
    edf_path: Path,
    xml_path: Optional[Path],
    channel_names: Optional[list[str]],
    target_sample_rate: float,
    input_channels: int,
    input_length: int,
    normalization: str = "zscore",
) -> torch.Tensor:
    """
    Load EDF, clip first wake episode based on XML, create bipolar C3-M2,
    resample (typically to 128 Hz), normalize, and interpolate/compress to
    fixed input_length.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Clip first full wake episode (first stage segment)
    if xml_path is not None and xml_path.exists():
        first_stage = parse_first_stage_from_xml(xml_path)
        if first_stage is not None:
            start, duration, stage = first_stage
            # Clip from start+duration onwards; first event is assumed wake
            clip_t0 = start + duration
            try:
                raw.crop(tmin=clip_t0, tmax=None)
                print_info(
                    f"[PREPROC] Clipped first stage ({stage}) of "
                    f"{duration:.1f}s from {edf_path.name}"
                )
            except Exception as e:
                print_warning(
                    f"[PREPROC] Failed to crop {edf_path.name} at {clip_t0:.1f}s: {e}"
                )

    # Create bipolar C3-M2 channel and combine with physio channels
    # This mirrors the behavior in data_extract_cfs.py
    try:
        ch_names = raw.ch_names
        if "C3" in ch_names and "M2" in ch_names:
            c3_signal = raw.copy().pick(["C3"]).get_data()[0]
            m2_signal = raw.copy().pick(["M2"]).get_data()[0]
            if len(c3_signal) != len(m2_signal):
                print_warning(
                    f"[PREPROC] C3 and M2 length mismatch in {edf_path.name}, "
                    "skipping bipolar fusion and using original channels."
                )
            else:
                # Bipolar C3-M2
                c3_m2_signal = (c3_signal - m2_signal).reshape(1, -1)
                sfreq = raw.info["sfreq"]

                # Physio channels: either from channel_names or default list
                default_physio = ["LOC", "ECG1", "EMG1", "THOR EFFORT", "ABDO EFFORT", "SaO2"]
                if channel_names:
                    physio_picks = [
                        ch
                        for ch in channel_names
                        if ch in default_physio and ch in ch_names
                    ]
                else:
                    physio_picks = [ch for ch in default_physio if ch in ch_names]

                if not physio_picks:
                    print_warning(
                        f"[PREPROC] No physio channels found in {edf_path.name}; "
                        "using only C3-M2."
                    )
                    combined_data = c3_m2_signal
                    combined_ch_names = ["C3-M2"]
                    ch_types = ["eeg"]
                else:
                    physio_raw = raw.copy().pick(physio_picks)
                    physio_data = physio_raw.get_data()
                    combined_data = np.vstack([c3_m2_signal, physio_data])
                    combined_ch_names = ["C3-M2"] + physio_picks
                    ch_types = ["eeg"] + ["misc"] * len(physio_picks)

                combined_info = mne.create_info(
                    ch_names=combined_ch_names,
                    sfreq=sfreq,
                    ch_types=ch_types,
                )
                raw = mne.io.RawArray(combined_data, combined_info)
        else:
            print_warning(
                f"[PREPROC] C3/M2 not both present in {edf_path.name}; "
                "skipping bipolar fusion and using original channels."
            )
    except Exception as e:
        print_warning(
            f"[PREPROC] Error creating C3-M2 bipolar channel for {edf_path.name}: {e}"
        )

    # Resample if needed (e.g., to 128 Hz)
    if target_sample_rate and abs(raw.info["sfreq"] - target_sample_rate) > 1e-3:
        raw.resample(target_sample_rate)

    # Channel selection (similar to CFSAilmentDataset._extract_channels)
    available = raw.ch_names
    if channel_names:
        picks = [ch for ch in channel_names if ch in available]
    else:
        picks = available[:input_channels]

    if not picks:
        raise RuntimeError(f"No matching channels found in EDF file: {edf_path}")

    data = raw.get_data(picks=picks)  # (channels, time)
    if data.shape[0] < input_channels:
        pad = np.zeros((input_channels - data.shape[0], data.shape[1]), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=0)
    elif data.shape[0] > input_channels:
        data = data[:input_channels]

    # Normalize (same as CFSAilmentDataset._normalize)
    if normalization == "zscore":
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-6
        data = (data - mean) / std
    elif normalization == "minmax":
        min_val = data.min(axis=1, keepdims=True)
        max_val = data.max(axis=1, keepdims=True)
        data = (data - min_val) / (max_val - min_val + 1e-6)

    # Convert to tensor and interpolate/compress to input_length
    tensor = torch.tensor(data, dtype=torch.float32)  # (C, T)
    current_len = tensor.shape[1]
    if current_len != input_length:
        tensor = tensor.unsqueeze(0)  # (1, C, T)
        tensor = F.interpolate(
            tensor, size=input_length, mode="linear", align_corners=False
        )
        tensor = tensor.squeeze(0)  # (C, input_length)

    return tensor


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CFS EDF signals into fixed-length tensors "
        "and rewrite CSV paths to .pt files."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(Path("csv-docs") / "cfs_visit5_selected.csv"),
        help="Path to CFS CSV file with 'path' column pointing to EDF files.",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="Directory containing CFS XML hypnogram files.",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default=str(Path("cfs_preprocessed")),
        help="Output directory for preprocessed .pt tensors.",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=8,
        help="Number of channels to keep (must match training config).",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=3000,
        help="Final sequence length (must match training config).",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=float,
        default=128.0,
        help="Target sampling rate for resampling (default: 128 Hz).",
    )
    parser.add_argument(
        "--channel_names",
        type=str,
        default="C3-M2,LOC,ECG1,EMG1,THOR EFFORT,ABDO EFFORT,SaO2",
        help="Comma-separated channel names to pick from EDF files "
        "(after C3-M2 bipolar fusion).",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["zscore", "minmax", "none"],
        default="zscore",
        help="Normalization strategy.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    annotation_dir = Path(args.annotation_dir)
    preprocessed_dir = Path(args.preprocessed_dir)

    if not csv_path.exists():
        print_error(f"CSV file not found: {csv_path}")
        return
    if not annotation_dir.exists():
        print_error(f"Annotation directory not found: {annotation_dir}")
        return

    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        print_error("CSV must contain a 'path' column.")
        return

    # Backup original CSV
    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not backup_path.exists():
        df.to_csv(backup_path, index=False)
        print_info(f"Backed up original CSV to {backup_path}")

    channel_list = [
        ch.strip()
        for ch in args.channel_names.split(",")
        if ch.strip()
    ] if args.channel_names else None

    new_paths = []
    num_rows = len(df)

    print_info(
        f"Starting preprocessing for {num_rows} rows. "
        f"Output directory: {preprocessed_dir}"
    )

    for idx, row in df.iterrows():
        edf_path = Path(row["path"])
        if not edf_path.exists():
            print_warning(f"[{idx+1}/{num_rows}] EDF not found, skipping: {edf_path}")
            new_paths.append(row["path"])
            continue

        # Determine corresponding XML
        xml_path = find_annotation_file(edf_path, annotation_dir)
        if xml_path is None:
            print_warning(
                f"[{idx+1}/{num_rows}] No XML found for {edf_path.name}, "
                f"skipping wake clipping for this file."
            )

        try:
            tensor = extract_and_preprocess_signal(
                edf_path=edf_path,
                xml_path=xml_path,
                channel_names=channel_list,
                target_sample_rate=args.target_sample_rate,
                input_channels=args.input_channels,
                input_length=args.input_length,
                normalization=args.normalization,
            )
        except Exception as e:
            print_warning(
                f"[{idx+1}/{num_rows}] Failed to preprocess {edf_path.name}: {e}"
            )
            new_paths.append(row["path"])
            continue

        # Save tensor
        nsrr_id = edf_path.stem.replace("cfs-visit5-", "")
        out_name = f"{nsrr_id}_preprocessed.pt"
        out_path = preprocessed_dir / out_name
        try:
            torch.save(tensor, out_path)
            new_paths.append(str(out_path))
            print_info(
                f"[{idx+1}/{num_rows}] Saved preprocessed tensor to {out_path}"
            )
        except Exception as e:
            print_warning(
                f"[{idx+1}/{num_rows}] Failed to save tensor for {edf_path.name}: {e}"
            )
            new_paths.append(row["path"])

    # Rewrite path column to point to preprocessed tensors (where available)
    df["path"] = new_paths
    df.to_csv(csv_path, index=False)
    print_success(f"Rewrote CSV in-place with new 'path' values: {csv_path}")


if __name__ == "__main__":
    main()


