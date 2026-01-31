from math import log2
import numpy as np

EPOCH_SEC = 30

def extract_features(annotations, lights_off=0):
    """
    Extract sleep architecture features from predicted sleep stage sequence.
    annotations: list[str] of sleep stages per epoch (W, N1, N2, N3, R)
    lights_off: seconds from recording start to lights-off
    """

    # ---- Normalize stages (merge N4 -> N3) ----
    stages = ['W', 'N1', 'N2', 'N3', 'R']
    ann = ['N3' if s == 'N4' else s for s in annotations]

    features = {}

    n_epochs = len(ann)
    total_recording_time = n_epochs * EPOCH_SEC

    # ---- Sleep onset ----
    sleep_epochs = [i for i, s in enumerate(ann) if s != 'W']
    if len(sleep_epochs) == 0:
        return None  # invalid night

    sleep_onset_idx = sleep_epochs[0]
    sleep_onset_sec = sleep_onset_idx * EPOCH_SEC

    # ---- Trim trailing wake after final sleep ----
    last_sleep_idx = max(sleep_epochs)
    effective_epochs = ann[sleep_onset_idx:last_sleep_idx + 1]

    # ---- Continuity & Efficiency ----
    total_sleep_epochs = sum(1 for s in effective_epochs if s != 'W')
    total_sleep_time = total_sleep_epochs * EPOCH_SEC

    time_in_bed = (last_sleep_idx + 1) * EPOCH_SEC - lights_off
    sleep_efficiency = total_sleep_time / time_in_bed if time_in_bed > 0 else 0

    wake_after_sleep_onset = effective_epochs.count('W') * EPOCH_SEC

    features.update({
        'Total_Sleep_Time': total_sleep_time,
        'Total_Time_In_Bed': time_in_bed,
        'Sleep_Efficiency': sleep_efficiency,
        'Sleep_Onset_Latency': sleep_onset_sec - lights_off,
        'Wake_After_Sleep_Onset': wake_after_sleep_onset
    })

    # ---- Stage Composition ----
    for s in stages:
        dur = ann.count(s) * EPOCH_SEC
        features[f'{s}_Duration'] = dur
        features[f'{s}_Proportion'] = dur / total_sleep_time if total_sleep_time > 0 else 0

    features['Light_Deep_Ratio'] = (
        (features['N1_Duration'] + features['N2_Duration']) /
        features['N3_Duration']
        if features['N3_Duration'] > 0 else 0
    )

    # ---- REM Latency & REM Periods ----
    try:
        rem_idx = ann.index('R')
        features['REM_Latency'] = (rem_idx - sleep_onset_idx) * EPOCH_SEC
    except ValueError:
        features['REM_Latency'] = -1  # no REM detected

    rem_periods = 0
    in_rem = False
    for s in effective_epochs:
        if s == 'R' and not in_rem:
            rem_periods += 1
            in_rem = True
        elif s != 'R':
            in_rem = False

    features['REM_Periods'] = rem_periods

    # ---- Transitions & Fragmentation ----
    transitions = 0
    transition_matrix = {a: {b: 0 for b in stages} for a in stages}

    for i in range(len(effective_epochs) - 1):
        if effective_epochs[i] != effective_epochs[i + 1]:
            transitions += 1
            transition_matrix[effective_epochs[i]][effective_epochs[i + 1]] += 1

    features['Stage_Transitions'] = transitions
    features['Transition_Rate'] = transitions / (total_recording_time / 3600)

    # ---- Transition Entropy ----
    entropy = 0
    if transitions > 0:
        for a in stages:
            for b in stages:
                p = transition_matrix[a][b] / transitions
                if p > 0:
                    entropy -= p * log2(p)

    features['Stage_Transition_Entropy'] = entropy

    # ---- Stage Segment Statistics ----
    segments = []
    current_stage = effective_epochs[0]
    length = 1

    for s in effective_epochs[1:]:
        if s == current_stage:
            length += 1
        else:
            segments.append((current_stage, length * EPOCH_SEC))
            current_stage = s
            length = 1
    segments.append((current_stage, length * EPOCH_SEC))

    seg_lengths = [l for _, l in segments]

    features['Mean_Segment_Duration'] = np.mean(seg_lengths)
    features['Segment_Duration_Variance'] = np.var(seg_lengths)
    features['Wake_Interruptions_per_Hour'] = (
        sum(1 for s, _ in segments if s == 'W') /
        (total_sleep_time / 3600)
    )

    return features
