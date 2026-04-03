"""
Full PSG Clinical Feature Extraction Pipeline
──────────────────────────────────────────────
Reads MESA-format EDF + XML files, computes clinical variables
across all major PSG domains, and writes one CSV row per patient.

Usage:
    python psg_feature_extraction.py \
        --edf_dir  /data/mesa/edfs \
        --xml_dir  /data/mesa/xmls \
        --out_dir  /data/mesa/features

Dependencies:
    pip install mne pyedflib scipy numpy pandas
"""

import argparse
import logging
import warnings
import xml.etree.ElementTree as ET
from math import log2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import scipy.signal as ssignal
from scipy.integrate import trapezoid

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("psg")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EPOCH_SEC = 30
FS_HIGH   = 256   # EEG, ECG, EMG, EOG, Pleth (original)
FS_LOW    = 32    # Airflow, Thor, Abdo, Leg EMG, Position (original)
FS_SPO2   = 1     # SpO2, HR (original – keep as-is)
FS_TARGET = 100   # Downsample high-rate channels to this

STAGES = ["W", "N1", "N2", "N3", "R"]

# EDF label → internal key  (add aliases as needed for your dataset)
CHANNEL_ALIASES: Dict[str, str] = {
    # ECG  – single channel
    "EKG": "ecg", "ECG": "ecg",
    # EOG  – left only (right is redundant for staging purposes)
    "E1": "eog_l", "EOG-L": "eog_l", "EOG(L)": "eog_l",
    # EMG (chin) – single channel
    "EMG": "emg_chin",
    # EEG  – EEG1 (Fz–Cz) only; EEG2/EEG3 intentionally excluded
    "EEG1": "eeg1", "EEG 1": "eeg1",
    # Respiratory – nasal-pressure flow only; thermistor / snore / aux excluded
    "Flow": "flow", "FLOW": "flow",
    # Thorax & abdomen effort (both needed to classify central vs obstructive)
    "Thor": "thor", "THOR": "thor",
    "Abdo": "abdo", "ABDO": "abdo",
    # Leg EMG – single combined channel
    "Leg": "leg_emg", "LEG": "leg_emg",
    # Oximetry – SpO2 only; Pleth / HR / DHR excluded
    "SpO2": "spo2", "SPO2": "spo2", "SaO2": "spo2",
    # Body position
    "Pos": "pos", "POS": "pos", "Position": "pos",
}

# NSRR XML stage concept → AASM stage
STAGE_CONCEPT_MAP: Dict[str, str] = {
    "Wake|0":            "W",
    "Stage 1 sleep|1":   "N1",
    "Stage 2 sleep|2":   "N2",
    "Stage 3 sleep|3":   "N3",
    "Stage 4 sleep|4":   "N3",   # merge N4 → N3
    "REM sleep|5":       "R",
    "Unscored|9":        "W",    # treat unscored as wake
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDF LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_edf(edf_path: Path) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Load EDF via MNE.  Returns {internal_key: (signal_array, fs)}.
    High-rate channels are anti-alias filtered then downsampled to FS_TARGET.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    channels: Dict[str, Tuple[np.ndarray, float]] = {}

    for ch_name in raw.ch_names:
        key = CHANNEL_ALIASES.get(ch_name)
        if key is None:
            # try case-insensitive
            key = CHANNEL_ALIASES.get(ch_name.strip())
        if key is None:
            continue  # skip unknown channels

        data, times = raw[ch_name]
        data = data.squeeze()
        fs_orig = raw.info["sfreq"]

        # Channels with original 256 Hz → downsample to 100 Hz
        if fs_orig >= 200:
            data = _downsample(data, int(fs_orig), FS_TARGET)
            fs = float(FS_TARGET)
        else:
            fs = float(fs_orig)

        channels[key] = (data, fs)
        log.debug(f"  Loaded {ch_name!r} → {key!r}  fs={fs:.0f} Hz  len={len(data)}")

    return channels


def _downsample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Anti-aliased downsample using polyphase filtering."""
    from math import gcd
    g = gcd(fs_in, fs_out)
    return ssignal.resample_poly(x, fs_out // g, fs_in // g)


# ─────────────────────────────────────────────────────────────────────────────
# 2. XML ANNOTATION PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_xml_stages(xml_path: Path) -> List[Tuple[float, float, str]]:
    """
    Parse full sleep-stage hypnogram from NSRR XML.
    Returns list of (start_sec, duration_sec, stage_code) sorted by start.
    """
    events = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for ev in root.findall(".//ScoredEvent"):
            type_el    = ev.find("EventType")
            concept_el = ev.find("EventConcept")
            start_el   = ev.find("Start")
            dur_el     = ev.find("Duration")

            if any(e is None for e in [type_el, concept_el, start_el, dur_el]):
                continue
            if "Stages|Stages" not in (type_el.text or ""):
                continue

            stage_code = STAGE_CONCEPT_MAP.get(concept_el.text or "", None)
            if stage_code is None:
                continue

            events.append((float(start_el.text), float(dur_el.text), stage_code))

    except Exception as e:
        log.warning(f"Failed to parse {xml_path}: {e}")

    return sorted(events, key=lambda x: x[0])


def build_epoch_sequence(
    stage_events: List[Tuple[float, float, str]],
    recording_duration_sec: float,
) -> List[str]:
    """Convert (start, duration, stage) events to per-epoch list."""
    n_epochs = int(np.ceil(recording_duration_sec / EPOCH_SEC))
    epoch_stages = ["W"] * n_epochs

    for start, dur, stage in stage_events:
        start_epoch = int(start // EPOCH_SEC)
        end_epoch   = int((start + dur) // EPOCH_SEC)
        for i in range(start_epoch, min(end_epoch, n_epochs)):
            epoch_stages[i] = stage

    return epoch_stages


def find_annotation_file(edf_path: Path, annotation_dir: Path) -> Optional[Path]:
    """Match EDF to NSRR XML by nsrrid."""
    stem = edf_path.stem
    if "mesa-sleep-" in stem:
        nsrrid = stem.replace("mesa-sleep-", "")
        candidate = annotation_dir / f"{nsrrid}-nsrr.xml"
        if candidate.exists():
            return candidate

    # Fallback: partial name match
    prefix = stem[:17]
    for f in annotation_dir.glob("*.xml"):
        if prefix in f.name:
            return f
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. SLEEP ARCHITECTURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_sleep_architecture(ann: List[str], lights_off: float = 0.0) -> Dict:
    """
    Compute standard polysomnographic sleep architecture features.
    ann       : per-epoch stage list (length = total recording epochs)
    lights_off: seconds from recording start to lights off
    """
    ann = ["N3" if s == "N4" else s for s in ann]
    features: Dict = {}
    n_epochs = len(ann)

    sleep_epochs = [i for i, s in enumerate(ann) if s != "W"]
    if not sleep_epochs:
        return {}

    sleep_onset_idx = sleep_epochs[0]
    last_sleep_idx  = sleep_epochs[-1]
    sleep_onset_sec = sleep_onset_idx * EPOCH_SEC

    effective = ann[sleep_onset_idx : last_sleep_idx + 1]

    total_sleep_epochs = sum(1 for s in effective if s != "W")
    total_sleep_time   = total_sleep_epochs * EPOCH_SEC
    time_in_bed        = (last_sleep_idx + 1) * EPOCH_SEC - lights_off
    sleep_efficiency   = total_sleep_time / time_in_bed if time_in_bed > 0 else 0
    waso               = effective.count("W") * EPOCH_SEC

    features.update(
        {
            "TST_sec":              total_sleep_time,
            "TIB_sec":              time_in_bed,
            "Sleep_Efficiency":     sleep_efficiency,
            "SOL_sec":              max(0, sleep_onset_sec - lights_off),
            "WASO_sec":             waso,
        }
    )

    # Stage durations & proportions
    for s in STAGES:
        dur = ann.count(s) * EPOCH_SEC
        features[f"{s}_Duration_sec"] = dur
        features[f"{s}_Proportion"]   = dur / total_sleep_time if total_sleep_time > 0 else 0

    features["N3_N2_Ratio"] = (
        features["N3_Duration_sec"] / features["N2_Duration_sec"]
        if features["N2_Duration_sec"] > 0 else 0
    )
    features["Light_Deep_Ratio"] = (
        (features["N1_Duration_sec"] + features["N2_Duration_sec"]) / features["N3_Duration_sec"]
        if features["N3_Duration_sec"] > 0 else 0
    )

    # REM latency
    try:
        rem_idx = ann.index("R")
        features["REM_Latency_sec"] = (rem_idx - sleep_onset_idx) * EPOCH_SEC
    except ValueError:
        features["REM_Latency_sec"] = -1

    # REM periods (discrete continuous blocks)
    rem_periods, in_rem = 0, False
    for s in effective:
        if s == "R" and not in_rem:
            rem_periods += 1; in_rem = True
        elif s != "R":
            in_rem = False
    features["REM_Periods"] = rem_periods

    # Transitions
    transitions = 0
    tm = {a: {b: 0 for b in STAGES} for a in STAGES}
    for i in range(len(effective) - 1):
        if effective[i] != effective[i + 1]:
            transitions += 1
            tm[effective[i]][effective[i + 1]] += 1

    features["Stage_Transitions"]   = transitions
    features["Transition_Rate_hr"]  = transitions / (total_sleep_time / 3600) if total_sleep_time > 0 else 0

    # Transition entropy
    ent = 0.0
    if transitions > 0:
        for a in STAGES:
            for b in STAGES:
                p = tm[a][b] / transitions
                if p > 0:
                    ent -= p * log2(p)
    features["Stage_Transition_Entropy"] = ent

    # Segment statistics
    segs = _segment_stages(effective)
    seg_durations = [d for _, d in segs]
    features["Mean_Segment_Duration_sec"]  = float(np.mean(seg_durations)) if seg_durations else 0
    features["Segment_Duration_Variance"]  = float(np.var(seg_durations))  if seg_durations else 0
    features["Wake_Interruptions_per_hr"]  = (
        sum(1 for s, _ in segs if s == "W") / (total_sleep_time / 3600)
        if total_sleep_time > 0 else 0
    )

    return features


def _segment_stages(stages: List[str]) -> List[Tuple[str, float]]:
    """Run-length encode a stage sequence → list of (stage, duration_sec)."""
    if not stages:
        return []
    segs, cur, length = [], stages[0], 1
    for s in stages[1:]:
        if s == cur:
            length += 1
        else:
            segs.append((cur, length * EPOCH_SEC))
            cur, length = s, 1
    segs.append((cur, length * EPOCH_SEC))
    return segs


def epoch_mask(ann: List[str], stage: str) -> np.ndarray:
    """Boolean array True for epochs matching stage."""
    return np.array([s == stage for s in ann])


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESPIRATORY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_respiratory_features(
    flow:    Optional[np.ndarray],
    thor:    Optional[np.ndarray],
    abdo:    Optional[np.ndarray],
    spo2:    Optional[np.ndarray],
    ann:     List[str],
    fs_flow: float = 32.0,
    fs_spo2: float = 1.0,
) -> Dict:
    """
    Compute AHI, AI, HI, CAI, OAI, REM-AHI, Supine-AHI,
    along with ODI, T90, T88, nadir SpO2, mean SpO2.
    """
    features: Dict = {}
    tst_hr = sum(1 for s in ann if s != "W") * EPOCH_SEC / 3600
    if tst_hr <= 0:
        return features

    # ── Airflow signal preprocessing ─────────────────────────────────────────
    apnea_events, hypopnea_events = [], []

    if flow is not None and len(flow) > 0:
        flow_filt = _bandpass(flow, 0.05, 4.0, fs_flow)
        envelope  = _moving_rms(flow_filt, int(fs_flow * 3))   # 3-sec window RMS
        baseline  = _moving_median(envelope, int(fs_flow * 60)) # 60-sec median baseline

        apnea_events    = _detect_flow_events(envelope, baseline, fs_flow, thresh=0.10, min_dur=10.0)
        hypopnea_events = _detect_flow_events(envelope, baseline, fs_flow, thresh=0.70, min_dur=10.0,
                                              upper_thresh=0.90)

    # ── Classify Central vs Obstructive via effort ───────────────────────────
    effort = None
    if thor is not None and abdo is not None:
        effort = np.abs(thor) + np.abs(abdo)   # summed respiratory effort

    obstructive_apneas, central_apneas, mixed_apneas = 0, 0, 0
    for start_s, end_s in apnea_events:
        if effort is not None:
            ev_effort = effort[int(start_s * fs_flow) : int(end_s * fs_flow)]
            mean_effort = float(np.mean(np.abs(ev_effort))) if len(ev_effort) else 0
            ev_baseline = float(np.mean(np.abs(effort))) * 0.2
            if mean_effort < ev_baseline:
                central_apneas += 1
            else:
                obstructive_apneas += 1
        else:
            obstructive_apneas += 1  # assume obstructive if no effort signal

    total_apneas    = len(apnea_events)
    total_hypopneas = len(hypopnea_events)
    total_events    = total_apneas + total_hypopneas

    features["AHI"]          = total_events    / tst_hr
    features["AI"]           = total_apneas    / tst_hr
    features["HI"]           = total_hypopneas / tst_hr
    features["OAI"]          = obstructive_apneas / tst_hr
    features["CAI"]          = central_apneas     / tst_hr
    features["Apnea_N_Total"] = total_apneas
    features["Hypopnea_N_Total"] = total_hypopneas

    if apnea_events:
        durations = [e - s for s, e in apnea_events]
        features["Apnea_Duration_Mean_sec"] = float(np.mean(durations))
        features["Apnea_Duration_Max_sec"]  = float(np.max(durations))
    else:
        features["Apnea_Duration_Mean_sec"] = 0.0
        features["Apnea_Duration_Max_sec"]  = 0.0

    # ── REM-AHI ──────────────────────────────────────────────────────────────
    rem_mask    = epoch_mask(ann, "R")
    rem_tst_hr  = float(np.sum(rem_mask)) * EPOCH_SEC / 3600
    if rem_tst_hr > 0 and flow is not None:
        rem_events = _filter_events_by_stage(apnea_events + hypopnea_events, rem_mask, EPOCH_SEC)
        features["REM_AHI"] = len(rem_events) / rem_tst_hr
    else:
        features["REM_AHI"] = 0.0

    # ── Oxygenation ──────────────────────────────────────────────────────────
    if spo2 is not None and len(spo2) > 0:
        oxy = extract_oxygenation_features(spo2, ann, fs_spo2)
        features.update(oxy)

    return features


def _bandpass(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    nyq = fs / 2
    lo_n, hi_n = lo / nyq, hi / nyq
    hi_n = min(hi_n, 0.999)
    b, a = ssignal.butter(4, [lo_n, hi_n], btype="band")
    return ssignal.filtfilt(b, a, x)


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    """Efficient moving-window RMS via cumulative sum of squares."""
    x2 = x ** 2
    cs = np.cumsum(x2)
    cs = np.concatenate([[0], cs])
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    pad = np.full(len(x) - len(rms), rms[0] if len(rms) else 0)
    return np.concatenate([pad, rms])


def _moving_median(x: np.ndarray, window: int) -> np.ndarray:
    """Approximate moving median using uniform_filter1d on absolute values."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x.astype(float), size=window, mode="nearest")


def _detect_flow_events(
    envelope:    np.ndarray,
    baseline:    np.ndarray,
    fs:          float,
    thresh:      float,
    min_dur:     float,
    upper_thresh: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Detect segments where envelope is below thresh*baseline (and above upper_thresh*baseline).
    Returns list of (start_sec, end_sec).
    """
    safe_baseline = np.where(baseline < 1e-6, 1e-6, baseline)
    ratio = envelope / safe_baseline
    mask  = (ratio < thresh) if upper_thresh >= 1.0 else ((ratio < thresh) & (ratio >= (1 - upper_thresh)))

    events = []
    in_event, start_i = False, 0
    for i, m in enumerate(mask):
        if m and not in_event:
            in_event, start_i = True, i
        elif not m and in_event:
            dur = (i - start_i) / fs
            if dur >= min_dur:
                events.append((start_i / fs, i / fs))
            in_event = False
    if in_event:
        dur = (len(mask) - start_i) / fs
        if dur >= min_dur:
            events.append((start_i / fs, len(mask) / fs))
    return events


def _filter_events_by_stage(
    events:     List[Tuple[float, float]],
    stage_mask: np.ndarray,   # bool array, one entry per epoch
    epoch_sec:  float,
) -> List[Tuple[float, float]]:
    """Keep only events whose midpoint falls in a True epoch."""
    kept = []
    for s, e in events:
        mid_epoch = int(((s + e) / 2) / epoch_sec)
        if mid_epoch < len(stage_mask) and stage_mask[mid_epoch]:
            kept.append((s, e))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# 5. OXYGENATION FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_oxygenation_features(
    spo2: np.ndarray,
    ann:  List[str],
    fs:   float = 1.0,
) -> Dict:
    features: Dict = {}

    # Clip physiological range
    spo2 = np.clip(spo2, 50, 100).astype(float)

    # Filter obvious artifacts (values == 0 or 255)
    valid = (spo2 > 50) & (spo2 < 101)
    spo2_clean = spo2[valid]

    if len(spo2_clean) == 0:
        return features

    features["SpO2_Mean"]   = float(np.mean(spo2_clean))
    features["SpO2_Nadir"]  = float(np.min(spo2_clean))
    features["SpO2_Std"]    = float(np.std(spo2_clean))

    total_sec = len(spo2) / fs
    features["T90_sec"]  = float(np.sum(spo2_clean < 90)) / fs
    features["T88_sec"]  = float(np.sum(spo2_clean < 88)) / fs
    features["T90_pct"]  = features["T90_sec"] / total_sec * 100 if total_sec > 0 else 0
    features["T88_pct"]  = features["T88_sec"] / total_sec * 100 if total_sec > 0 else 0

    # ODI: desaturation events ≥3% dip
    features["ODI3"]  = _compute_odi(spo2_clean, fs, drop_pct=3.0)
    features["ODI4"]  = _compute_odi(spo2_clean, fs, drop_pct=4.0)

    # Desaturation AUC (area under 90%)
    deficit = np.where(spo2_clean < 90, 90 - spo2_clean, 0)
    features["Desaturation_AUC"] = float(trapezoid(deficit)) / fs

    return features


def _compute_odi(spo2: np.ndarray, fs: float, drop_pct: float = 3.0) -> float:
    """Count ≥drop_pct desaturation events per hour of recording."""
    total_hrs = len(spo2) / fs / 3600
    if total_hrs <= 0:
        return 0.0

    events, in_desat, peak = 0, False, 0.0
    for val in spo2:
        if not in_desat:
            peak = val
            if val < peak - drop_pct:
                in_desat = True
                events += 1
        else:
            if val >= peak - (drop_pct / 2):  # recovery threshold
                in_desat = False
    return events / total_hrs


# ─────────────────────────────────────────────────────────────────────────────
# 6. EEG SPECTRAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
FREQ_BANDS = {
    "Delta":  (0.5,  4.0),
    "Theta":  (4.0,  8.0),
    "Alpha":  (8.0, 12.0),
    "Sigma": (12.0, 16.0),   # spindle band
    "Beta":  (16.0, 30.0),
}


def extract_eeg_features(
    eeg:    np.ndarray,
    ann:    List[str],
    fs:     float = 100.0,
    ch_name: str = "eeg3",
) -> Dict:
    """
    Per-stage spectral power, alpha intrusion in NREM,
    beta hyperarousal index, spindle-band power, spectral entropy.
    eeg : 1-D array of the full recording
    ann : per-epoch stage list
    """
    features: Dict = {}
    epoch_len_s = int(fs * EPOCH_SEC)
    n_epochs    = min(len(ann), len(eeg) // epoch_len_s)

    band_powers_by_stage: Dict[str, Dict[str, List[float]]] = {
        st: {b: [] for b in FREQ_BANDS} for st in STAGES
    }
    all_ents: Dict[str, List[float]] = {st: [] for st in STAGES}

    for i in range(n_epochs):
        stage = ann[i]
        seg   = eeg[i * epoch_len_s : (i + 1) * epoch_len_s]
        if len(seg) < epoch_len_s:
            continue

        # Hann-windowed Welch PSD
        freqs, psd = ssignal.welch(seg, fs=fs, nperseg=int(fs * 4), window="hann")

        # Band powers
        for band, (lo, hi) in FREQ_BANDS.items():
            idx = (freqs >= lo) & (freqs < hi)
            power = float(trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0
            band_powers_by_stage[stage][band].append(power)

        # Spectral entropy (Shannon over PSD)
        psd_norm = psd / (psd.sum() + 1e-12)
        ent = -float(np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
        all_ents[stage].append(ent)

    prefix = ch_name.upper()

    for st in STAGES:
        for band in FREQ_BANDS:
            vals = band_powers_by_stage[st][band]
            features[f"{prefix}_{st}_{band}_Power_mean"] = float(np.mean(vals)) if vals else 0.0
            features[f"{prefix}_{st}_{band}_Power_std"]  = float(np.std(vals))  if vals else 0.0

        ents = all_ents[st]
        features[f"{prefix}_{st}_SpectralEntropy_mean"] = float(np.mean(ents)) if ents else 0.0

    # Alpha intrusion in NREM: mean alpha / delta in N2+N3
    for st in ["N2", "N3"]:
        a_vals = band_powers_by_stage[st]["Alpha"]
        d_vals = band_powers_by_stage[st]["Delta"]
        if a_vals and d_vals:
            ratio = [a / (d + 1e-12) for a, d in zip(a_vals, d_vals)]
            features[f"{prefix}_{st}_AlphaIntrusion"] = float(np.mean(ratio))
        else:
            features[f"{prefix}_{st}_AlphaIntrusion"] = 0.0

    # Beta hyperarousal: mean NREM beta across N1+N2+N3
    nrem_beta = []
    for st in ["N1", "N2", "N3"]:
        nrem_beta.extend(band_powers_by_stage[st]["Beta"])
    features[f"{prefix}_NREM_Beta_Hyperarousal"] = float(np.mean(nrem_beta)) if nrem_beta else 0.0

    # Sigma (spindle) power during N2
    n2_sigma = band_powers_by_stage["N2"]["Sigma"]
    features[f"{prefix}_N2_Sigma_mean"] = float(np.mean(n2_sigma)) if n2_sigma else 0.0

    # Delta ratio (N3 delta / waking delta)
    w_delta = band_powers_by_stage["W"]["Delta"]
    n3_delta = band_powers_by_stage["N3"]["Delta"]
    features[f"{prefix}_Delta_SWA_ratio"] = (
        float(np.mean(n3_delta)) / (float(np.mean(w_delta)) + 1e-12)
        if n3_delta and w_delta else 0.0
    )

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 7. CARDIAC / HRV FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_cardiac_features(
    ecg: np.ndarray,
    ann: List[str],
    fs:  float = 100.0,
) -> Dict:
    """
    SDNN, RMSSD, LF power, HF power, LF/HF ratio, mean HR.
    Uses a simple derivative-based R-peak detector; upgrade to
    neurokit2 / biosppy for clinical-grade results.
    """
    features: Dict = {}
    if ecg is None or len(ecg) == 0:
        return features

    rpeaks = _detect_rpeaks(ecg, fs)
    if len(rpeaks) < 10:
        return features

    rr_sec = np.diff(rpeaks) / fs   # RR intervals in seconds

    # Time-domain HRV
    features["HRV_SDNN_ms"]   = float(np.std(rr_sec)   * 1000)
    features["HRV_RMSSD_ms"]  = float(np.sqrt(np.mean(np.diff(rr_sec) ** 2)) * 1000)
    features["HRV_MeanRR_ms"] = float(np.mean(rr_sec)  * 1000)
    features["HRV_MeanHR_bpm"] = 60.0 / float(np.mean(rr_sec)) if np.mean(rr_sec) > 0 else 0

    # Frequency-domain HRV via Welch on uniformly resampled RR series
    try:
        lf, hf, lf_hf = _hrv_frequency_domain(rr_sec, rpeaks / fs)
        features["HRV_LF_ms2"]    = lf
        features["HRV_HF_ms2"]    = hf
        features["HRV_LF_HF"]     = lf_hf
    except Exception:
        pass

    # Stage-specific mean HR
    ecg_epoch_len = int(fs * EPOCH_SEC)
    for st in STAGES:
        hr_vals = []
        for i, stage in enumerate(ann):
            if stage != st:
                continue
            seg_rr = rr_sec[(rpeaks[:-1] >= i * ecg_epoch_len) & (rpeaks[:-1] < (i + 1) * ecg_epoch_len)]
            if len(seg_rr) > 0:
                hr_vals.append(60.0 / float(np.mean(seg_rr)))
        features[f"MeanHR_{st}_bpm"] = float(np.mean(hr_vals)) if hr_vals else 0.0

    return features


def _detect_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Simple Pan-Tompkins-inspired R-peak detector."""
    # Bandpass 5–15 Hz
    nyq = fs / 2
    b, a = ssignal.butter(2, [5 / nyq, 15 / nyq], btype="band")
    filtered = ssignal.filtfilt(b, a, ecg)

    # Differentiate & square
    diff_sq = np.diff(filtered) ** 2

    # Moving average integration (~150 ms window)
    window = int(0.15 * fs)
    integrated = np.convolve(diff_sq, np.ones(window) / window, mode="same")

    # Find peaks with minimum distance (~400 ms)
    min_dist = int(0.4 * fs)
    threshold = 0.5 * np.max(integrated)
    peaks, _ = ssignal.find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks


def _hrv_frequency_domain(
    rr_sec: np.ndarray, peak_times: np.ndarray
) -> Tuple[float, float, float]:
    """Interpolate RR series to 4 Hz then compute LF/HF via Welch."""
    from scipy.interpolate import interp1d

    t_rr   = peak_times[:-1] + np.diff(peak_times) / 2   # midpoint times
    fs_itp = 4.0
    t_itp  = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_itp)

    f_itp  = interp1d(t_rr, rr_sec, kind="cubic", fill_value="extrapolate")
    rr_itp = f_itp(t_itp) * 1000  # convert to ms

    # Detrend
    rr_itp = ssignal.detrend(rr_itp)

    freqs, psd = ssignal.welch(rr_itp, fs=fs_itp, nperseg=min(256, len(rr_itp)))

    lf_idx = (freqs >= 0.04) & (freqs < 0.15)
    hf_idx = (freqs >= 0.15) & (freqs < 0.40)

    lf = float(trapezoid(psd[lf_idx], freqs[lf_idx])) if np.any(lf_idx) else 0.0
    hf = float(trapezoid(psd[hf_idx], freqs[hf_idx])) if np.any(hf_idx) else 0.0
    lf_hf = lf / hf if hf > 1e-9 else 0.0

    return lf, hf, lf_hf


# ─────────────────────────────────────────────────────────────────────────────
# 8. PERIODIC LIMB MOVEMENT (PLM) FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_plm_features(
    leg_emg: np.ndarray,
    ann:     List[str],
    fs:      float = 32.0,
) -> Dict:
    """
    AASM PLM criteria:
    - Burst duration: 0.5 – 10 sec
    - Amplitude: ≥8× baseline
    - Inter-movement interval: 5 – 90 sec
    - ≥4 consecutive qualifying movements = PLM series
    """
    features: Dict = {}
    if leg_emg is None or len(leg_emg) == 0:
        return features

    tst_hr = sum(1 for s in ann if s != "W") * EPOCH_SEC / 3600
    if tst_hr <= 0:
        return features

    # Rectify & smooth
    env = np.abs(leg_emg)
    smooth_win = max(1, int(fs * 0.1))  # 100 ms
    env = np.convolve(env, np.ones(smooth_win) / smooth_win, mode="same")
    baseline = float(np.percentile(env, 10))

    # Detect bursts: amplitude ≥ 8× baseline, duration 0.5–10 sec
    threshold = max(baseline * 8, np.std(env) * 2)
    above     = env > threshold

    bursts: List[Tuple[float, float]] = []
    in_burst, start_i = False, 0
    for i, a in enumerate(above):
        if a and not in_burst:
            in_burst, start_i = True, i
        elif not a and in_burst:
            dur = (i - start_i) / fs
            if 0.5 <= dur <= 10.0:
                bursts.append((start_i / fs, i / fs))
            in_burst = False

    # PLM series: ≥4 consecutive bursts with IMI 5–90 sec
    plm_count = 0
    if len(bursts) >= 4:
        imi_list = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
        valid_series = [5.0 <= imi <= 90.0 for imi in imi_list]

        # Count qualifying PLMs in runs of ≥4
        run = 0
        for v in valid_series:
            if v:
                run += 1
                if run >= 3:   # 3 valid gaps = 4 movements
                    plm_count += 1
            else:
                run = 0

    features["PLM_N"]          = plm_count
    features["PLMI"]           = plm_count / tst_hr
    features["Leg_Burst_N"]    = len(bursts)
    features["Leg_Burst_Rate"] = len(bursts) / tst_hr

    if len(bursts) > 1:
        imis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
        features["IMI_Mean_sec"] = float(np.mean(imis))
        features["IMI_Std_sec"]  = float(np.std(imis))
    else:
        features["IMI_Mean_sec"] = 0.0
        features["IMI_Std_sec"]  = 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 9. REM BEHAVIOR DISORDER / EMG FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_rem_emg_features(
    emg_chin: np.ndarray,
    ann:      List[str],
    fs:       float = 100.0,
) -> Dict:
    """
    RBD metrics: tonic and phasic REM EMG activity.
    AASM thresholds: tonic = >30% of REM epoch above 2× NREM baseline
                     phasic = bursts 0.1–5 sec above 4× NREM baseline
    """
    features: Dict = {}
    if emg_chin is None or len(emg_chin) == 0:
        return features

    epoch_len = int(fs * EPOCH_SEC)
    n_epochs  = min(len(ann), len(emg_chin) // epoch_len)

    nrem_rms_vals = []
    for i in range(n_epochs):
        if ann[i] in ("N1", "N2", "N3"):
            seg = emg_chin[i * epoch_len : (i + 1) * epoch_len]
            nrem_rms_vals.append(float(np.sqrt(np.mean(seg ** 2))))
    nrem_baseline = float(np.median(nrem_rms_vals)) if nrem_rms_vals else 1e-6

    tonic_thresh  = 2.0 * nrem_baseline
    phasic_thresh = 4.0 * nrem_baseline

    rem_epochs_n, tonic_rem_n, phasic_burst_count = 0, 0, 0

    for i in range(n_epochs):
        if ann[i] != "R":
            continue
        rem_epochs_n += 1
        seg = emg_chin[i * epoch_len : (i + 1) * epoch_len]
        env = np.abs(seg)

        # Tonic: >30% of epoch above threshold
        if np.mean(env > tonic_thresh) > 0.30:
            tonic_rem_n += 1

        # Phasic: bursts 0.1–5 sec above threshold
        above = env > phasic_thresh
        in_b, start_i = False, 0
        for j, a in enumerate(above):
            if a and not in_b:
                in_b, start_i = True, j
            elif not a and in_b:
                dur = (j - start_i) / fs
                if 0.1 <= dur <= 5.0:
                    phasic_burst_count += 1
                in_b = False

    features["REM_Epochs_N"]               = rem_epochs_n
    features["REM_Tonic_EMG_Epochs_N"]     = tonic_rem_n
    features["REM_Tonic_EMG_Pct"]          = (tonic_rem_n / rem_epochs_n * 100) if rem_epochs_n > 0 else 0.0
    features["REM_Phasic_EMG_Bursts_N"]    = phasic_burst_count
    features["REM_Phasic_EMG_Rate_per_hr"] = (
        phasic_burst_count / (rem_epochs_n * EPOCH_SEC / 3600) if rem_epochs_n > 0 else 0.0
    )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 10. BRUXISM FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def extract_bruxism_features(
    emg_chin: np.ndarray,
    ann:      List[str],
    fs:       float = 100.0,
) -> Dict:
    """
    Bruxism Episode Index (BEI), burst characteristics.
    Criteria: bursts 0.25–2 sec, ≥4× background RMS.
    """
    features: Dict = {}
    if emg_chin is None or len(emg_chin) == 0:
        return features

    tst_hr = sum(1 for s in ann if s != "W") * EPOCH_SEC / 3600

    env       = np.abs(emg_chin)
    background = float(np.percentile(env, 20))
    threshold  = 4.0 * background

    bursts: List[float] = []
    in_b, start_i = False, 0
    for i, v in enumerate(env):
        if v > threshold and not in_b:
            in_b, start_i = True, i
        elif v <= threshold and in_b:
            dur = (i - start_i) / fs
            if 0.25 <= dur <= 2.0:
                bursts.append(dur)
            in_b = False

    features["Bruxism_Burst_N"]         = len(bursts)
    features["BEI"]                     = len(bursts) / tst_hr if tst_hr > 0 else 0.0
    features["Bruxism_Burst_Dur_mean"]  = float(np.mean(bursts)) if bursts else 0.0
    features["Bruxism_Burst_Dur_std"]   = float(np.std(bursts))  if bursts else 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 11. PATIENT-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_patient(
    edf_path:       Path,
    xml_path:       Path,
    output_dir:     Path,
    lights_off_sec: float = 0.0,
) -> Optional[Dict]:
    """
    Full feature extraction for one patient.
    Saves features to output_dir/<nsrrid>_features.csv
    Returns the feature dict (or None on failure).
    """
    nsrrid = edf_path.stem.replace("mesa-sleep-", "")
    log.info(f"Processing {nsrrid} ...")

    # ── Load EDF ─────────────────────────────────────────────────────────────
    try:
        channels = load_edf(edf_path)
    except Exception as e:
        log.error(f"EDF load failed for {edf_path}: {e}")
        return None

    # ── Parse XML stages ─────────────────────────────────────────────────────
    stage_events = parse_xml_stages(xml_path)
    if not stage_events:
        log.warning(f"No stage events found in {xml_path}")
        return None

    # Determine recording duration from EDF
    # Use any available channel length
    rec_dur = 0.0
    for key, (data, fs) in channels.items():
        rec_dur = max(rec_dur, len(data) / fs)
    if rec_dur == 0:
        log.error(f"Could not determine recording duration for {nsrrid}")
        return None

    ann = build_epoch_sequence(stage_events, rec_dur)

    # Helper to get channel safely
    def ch(name: str):
        return channels.get(name, (None, 1.0))

    # ── Feature extraction ───────────────────────────────────────────────────
    all_features: Dict = {"nsrrid": nsrrid}

    # 1. Sleep architecture
    arch = extract_sleep_architecture(ann, lights_off=lights_off_sec)
    all_features.update(arch)

    # 2. Respiratory + Oxygenation
    flow_data, fs_flow = ch("flow")
    thor_data, _       = ch("thor")
    abdo_data, _       = ch("abdo")
    spo2_data, fs_spo2 = ch("spo2")

    resp = extract_respiratory_features(
        flow    = flow_data,
        thor    = thor_data,
        abdo    = abdo_data,
        spo2    = spo2_data,
        ann     = ann,
        fs_flow = float(FS_LOW),
        fs_spo2 = float(FS_SPO2),
    )
    all_features.update(resp)

    # 3. EEG spectral – EEG1 (Fz–Cz) only
    eeg_data, _ = ch("eeg1")
    if eeg_data is not None:
        eeg_feats = extract_eeg_features(eeg_data, ann, fs=float(FS_TARGET), ch_name="eeg1")
        all_features.update(eeg_feats)

    # 4. Cardiac HRV
    ecg_data, fs_ecg = ch("ecg")
    if ecg_data is not None:
        cardiac = extract_cardiac_features(ecg_data, ann, fs=float(FS_TARGET))
        all_features.update(cardiac)

    # 5. PLM
    leg_data, fs_leg = ch("leg_emg")
    if leg_data is not None:
        plm = extract_plm_features(leg_data, ann, fs=float(FS_LOW))
        all_features.update(plm)

    # 6. REM EMG / RBD
    emg_data, fs_emg = ch("emg_chin")
    if emg_data is not None:
        rbd = extract_rem_emg_features(emg_data, ann, fs=float(FS_TARGET))
        all_features.update(rbd)
        bru = extract_bruxism_features(emg_data, ann, fs=float(FS_TARGET))
        all_features.update(bru)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{nsrrid}_features.csv"
    pd.DataFrame([all_features]).to_csv(out_path, index=False)
    log.info(f"  Saved {len(all_features)} features → {out_path}")

    return all_features


# ─────────────────────────────────────────────────────────────────────────────
# 12. BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(
    edf_dir:    Path,
    xml_dir:    Path,
    output_dir: Path,
    n_jobs:     int = 1,
) -> pd.DataFrame:
    """
    Process all EDF files in edf_dir, match to XMLs in xml_dir,
    write per-patient CSVs and a combined summary CSV.
    """
    edf_files = sorted(edf_dir.glob("*.edf"))
    log.info(f"Found {len(edf_files)} EDF files in {edf_dir}")

    all_records = []

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for edf_path in edf_files:
                xml_path = find_annotation_file(edf_path, xml_dir)
                if xml_path is None:
                    log.warning(f"No XML found for {edf_path.name}; skipping.")
                    continue
                f = ex.submit(process_patient, edf_path, xml_path, output_dir)
                futures[f] = edf_path.stem

            for f in as_completed(futures):
                result = f.result()
                if result:
                    all_records.append(result)
    else:
        for edf_path in edf_files:
            xml_path = find_annotation_file(edf_path, xml_dir)
            if xml_path is None:
                log.warning(f"No XML found for {edf_path.name}; skipping.")
                continue
            result = process_patient(edf_path, xml_path, output_dir)
            if result:
                all_records.append(result)

    # Combine all patients into one summary CSV
    if all_records:
        summary = pd.DataFrame(all_records)
        summary_path = output_dir / "all_patients_features.csv"
        summary.to_csv(summary_path, index=False)
        log.info(f"\nSummary CSV ({len(summary)} patients, {len(summary.columns)} features)"
                 f" → {summary_path}")
        return summary

    log.warning("No records processed.")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PSG Clinical Feature Extraction")
    parser.add_argument("--edf_dir",  type=Path, required=True, help="Directory with .edf files")
    parser.add_argument("--xml_dir",  type=Path, required=True, help="Directory with NSRR XML files")
    parser.add_argument("--out_dir",  type=Path, required=True, help="Output directory for CSV files")
    parser.add_argument("--n_jobs",   type=int,  default=1,     help="Parallel workers (default: 1)")
    args = parser.parse_args()

    run_batch(args.edf_dir, args.xml_dir, args.out_dir, args.n_jobs)


if __name__ == "__main__":
    main()
