"""
Offline preprocessing for MESA EDF signals.

Steps:
- Load EDF using MNE
- Find corresponding XML hypnogram
- Clip the first full wake episode (first sleep stage segment)
- Select channels: EEG1, EEG2, EEG3, Thor
- Resample EEG channels to 128 Hz (Thor keeps original sampling rate)
- Normalize per channel
- Save as .pt tensor
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional, Tuple

import mne
import numpy as np
import torch
import xml.etree.ElementTree as ET

from utils import print_info, print_warning, print_success, print_error


def find_annotation_file(edf_path: Path, annotation_dir: Path) -> Optional[Path]:
    """
    Find corresponding XML annotation file for a MESA EDF file.

    Naming convention:
    - EDF:  mesa-sleep-{nsrrid}.edf
    - XML:  {nsrrid}-nsrr.xml
    """
    base_name = edf_path.stem
    # Primary pattern
    if "mesa-sleep-" in base_name:
        nsrrid = base_name.replace("mesa-sleep-", "")
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

    MESA XML files store events in <ScoredEvent> with:
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
    target_eeg_fs: float = 128.0,
    normalization: str = "zscore",
) -> torch.Tensor:
    """
    Load EDF, clip first wake episode based on XML, select channels (EEG1, EEG2, EEG3, Thor),
    resample EEG channels to target_fs, normalize, and return as tensor.

    Args:
        edf_path: Path to EDF file
        xml_path: Optional path to XML annotation file
        target_eeg_fs: Target sampling rate for EEG channels (default: 128 Hz)
        normalization: Normalization method ('zscore' or 'minmax')

    Returns:
        Tensor of shape (4, T) where T is the length after resampling/interpolation
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Clip first full wake episode (first stage segment)
    if xml_path is not None and xml_path.exists():
        first_stage = parse_first_stage_from_xml(xml_path)
        if first_stage is not None:
            start, duration, stage = first_stage
            if "Wake" in stage or "wake" in stage.lower():
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

    # Select channels: EEG1, EEG2, EEG3, Thor
    required_channels = ["EEG1", "EEG2", "EEG3", "Thor"]
    available_channels = raw.ch_names

    # Check which channels are available
    eeg_channels = [ch for ch in ["EEG1", "EEG2", "EEG3"] if ch in available_channels]
    thor_channel = "Thor" if "Thor" in available_channels else None

    if not eeg_channels:
        raise RuntimeError(
            f"No EEG channels (EEG1, EEG2, EEG3) found in EDF file: {edf_path}"
        )

    # Process EEG channels
    eeg_data_list = []
    eeg_sfreq = None

    for eeg_ch in eeg_channels:
        eeg_raw = raw.copy().pick([eeg_ch])
        eeg_data = eeg_raw.get_data()[0]  # (time,)
        eeg_sfreq = eeg_raw.info["sfreq"]

        # Resample EEG to target_fs if needed
        if abs(eeg_sfreq - target_eeg_fs) > 1e-3:
            eeg_raw_resampled = eeg_raw.copy().resample(target_eeg_fs)
            eeg_data = eeg_raw_resampled.get_data()[0]
            eeg_sfreq = target_eeg_fs

        eeg_data_list.append(eeg_data)

    # Process Thor channel (resample to match EEG sampling rate)
    if thor_channel:
        thor_raw = raw.copy().pick([thor_channel])
        thor_data = thor_raw.get_data()[0]  # (time,)
        thor_sfreq = thor_raw.info["sfreq"]

        # Resample Thor to match EEG sampling rate
        if abs(thor_sfreq - eeg_sfreq) > 1e-3:
            # Create temporary RawArray for resampling
            thor_info = mne.create_info(
                ch_names=[thor_channel], sfreq=thor_sfreq, ch_types=["resp"]
            )
            thor_raw_array = mne.io.RawArray(
                thor_data.reshape(1, -1), thor_info
            )
            thor_raw_resampled = thor_raw_array.resample(eeg_sfreq)
            thor_data = thor_raw_resampled.get_data()[0]
    else:
        print_warning(
            f"[PREPROC] Thor channel not found in {edf_path.name}, "
            "using zeros for Thor channel."
        )
        # Create zero array matching EEG length (use first EEG channel as reference)
        if eeg_data_list:
            thor_data = np.zeros_like(eeg_data_list[0])
        else:
            raise RuntimeError("No EEG channels available to determine length for Thor channel")

    # Stack all channels: (4, time)
    # Pad missing EEG channels with zeros if needed
    while len(eeg_data_list) < 3:
        eeg_data_list.append(np.zeros_like(eeg_data_list[0]))

    # Ensure all channels have the same length
    min_length = min(
        len(eeg_data_list[0]),
        len(eeg_data_list[1]),
        len(eeg_data_list[2]),
        len(thor_data),
    )

    channel_data = np.array(
        [
            eeg_data_list[0][:min_length],
            eeg_data_list[1][:min_length],
            eeg_data_list[2][:min_length],
            thor_data[:min_length],
        ]
    )  # (4, time)

    # Normalize per channel
    if normalization == "zscore":
        mean = channel_data.mean(axis=1, keepdims=True)
        std = channel_data.std(axis=1, keepdims=True) + 1e-6
        channel_data = (channel_data - mean) / std
    elif normalization == "minmax":
        min_val = channel_data.min(axis=1, keepdims=True)
        max_val = channel_data.max(axis=1, keepdims=True)
        channel_data = (channel_data - min_val) / (max_val - min_val + 1e-6)

    # Convert to tensor
    tensor = torch.tensor(channel_data, dtype=torch.float32)  # (4, T)

    return tensor


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MESA EDF signals into tensors with channels: EEG1, EEG2, EEG3, Thor."
    )
    parser.add_argument(
        "--edf_folder",
        type=str,
        required=True,
        help="Directory containing MESA EDF files.",
    )
    parser.add_argument(
        "--annotation_folder",
        type=str,
        required=True,
        help="Directory containing MESA XML hypnogram files.",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default=str(Path("MESA_preprocessed")),
        help="Output directory for preprocessed .pt tensors.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="If set, regenerate tensors even if an output .pt already exists.",
    )
    parser.add_argument(
        "--target_eeg_fs",
        type=float,
        default=128.0,
        help="Target sampling rate for EEG channels (default: 128 Hz).",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["zscore", "minmax", "none"],
        default="zscore",
        help="Normalization strategy.",
    )

    args = parser.parse_args()

    edf_folder = Path(args.edf_folder)
    annotation_folder = Path(args.annotation_folder)
    preprocessed_dir = Path(args.preprocessed_dir)

    if not edf_folder.exists():
        print_error(f"EDF folder not found: {edf_folder}")
        return
    if not annotation_folder.exists():
        print_error(f"Annotation folder not found: {annotation_folder}")
        return

    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Find all EDF files
    edf_files = list(edf_folder.glob("*.edf"))
    if not edf_files:
        print_warning(f"No EDF files found in {edf_folder}")
        return

    num_files = len(edf_files)
    print_info(
        f"Starting preprocessing for {num_files} EDF files. "
        f"Output directory: {preprocessed_dir}"
    )

    successful = 0
    failed = 0

    for idx, edf_path in enumerate(edf_files):
        # Determine corresponding XML
        xml_path = find_annotation_file(edf_path, annotation_folder)
        if xml_path is None:
            print_warning(
                f"[{idx+1}/{num_files}] No XML found for {edf_path.name}, "
                f"skipping wake clipping for this file."
            )

        try:
            tensor = extract_and_preprocess_signal(
                edf_path=edf_path,
                xml_path=xml_path,
                target_eeg_fs=args.target_eeg_fs,
                normalization=args.normalization,
            )
        except Exception as e:
            print_warning(
                f"[{idx+1}/{num_files}] Failed to preprocess {edf_path.name}: {e}"
            )
            failed += 1
            continue

        # Save tensor (optionally overwrite)
        out_name = f"{edf_path.stem}_preprocessed.pt"
        out_path = preprocessed_dir / out_name
        try:
            if out_path.exists() and not args.overwrite_existing:
                print_info(
                    f"[{idx+1}/{num_files}] Exists, reusing {out_path} "
                    "(use --overwrite_existing to rebuild)"
                )
            else:
                torch.save(tensor, out_path)
                print_info(
                    f"[{idx+1}/{num_files}] Saved preprocessed tensor to {out_path} "
                    f"(shape: {tensor.shape})"
                )
            successful += 1
        except Exception as e:
            print_warning(
                f"[{idx+1}/{num_files}] Failed to save tensor for {edf_path.name}: {e}"
            )
            failed += 1

    print_success(
        f"Preprocessing complete: {successful} successful, {failed} failed. "
        f"Output directory: {preprocessed_dir}"
    )


if __name__ == "__main__":
    main()
