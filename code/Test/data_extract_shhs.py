from pathlib import Path

import numpy as np
import mne

# Configure your input and output directories here
input_dir = Path(r'F:\Sleep-Staging\test')
output_dir = Path(r'F:\Sleep-Staging\result')
output_dir.mkdir(parents=True, exist_ok=True)

# Select other physiological channels to keep
# Adjust channel names as needed based on your EDF
physio_channels = ['EEG(sec)','EOG(L)','ECG','EMG','THOR RES', 'ABDO RES', 'SaO2']
target_sfreq = 128  # Hz
MAX_EDF_DIGITS = 8
PHYS_RANGE_LIMIT = 10 ** (MAX_EDF_DIGITS - 1) - 1


def scale_to_edf_range(inst, limit=PHYS_RANGE_LIMIT):
    """Scale channels so their physical range fits within EDF constraints."""
    if not inst.preload:
        inst.load_data()

    data = inst.get_data()
    scaled_channels = {}

    for ch_idx, ch_name in enumerate(inst.ch_names):
        peak = float(np.max(np.abs(data[ch_idx])))
        if peak == 0 or peak <= limit:
            continue

        scale_exp = int(np.ceil(np.log10(peak)) - np.log10(limit))
        scale_factor = 10 ** scale_exp
        data[ch_idx] /= scale_factor
        scaled_channels[ch_name] = scale_factor

    if scaled_channels:
        inst._data = data

    return scaled_channels

for edf_path in sorted(input_dir.glob('*.edf')):
    print(f'Processing {edf_path.name}...')
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)

    # Pick the available physio channels from original data
    physio_picks = [ch for ch in physio_channels if ch in raw.ch_names]
    if not physio_picks:
        print('  No matching physio channels found; skipping file.')
        continue

    physio_raw = raw.copy().pick(physio_picks)

    # Resample all channels to the target sampling frequency
    physio_raw.resample(target_sfreq, npad='auto')

    scaled = scale_to_edf_range(physio_raw)
    if scaled:
        print('  Applied scaling factors (data divided by factor):')
        for ch, factor in scaled.items():
            print(f'    {ch}: 1/{factor}')

    # Save combined data to new EDF file
    output_file = output_dir / f'{edf_path.stem}_combined.edf'
    physio_raw.export(
        str(output_file),
        fmt='edf',
        physical_range=(-PHYS_RANGE_LIMIT, PHYS_RANGE_LIMIT),
    )

    print(f'  Combined C3-M2 and physiological signals saved to {output_file}')