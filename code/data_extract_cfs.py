from pathlib import Path

import numpy as np
import mne

# Configure your input and output directories here
input_dir = Path(r'D:\cfs\polysomnography\edfs')
output_dir = Path(r'D:\cfs\polysomnography\edfs_combined')
output_dir.mkdir(parents=True, exist_ok=True)

# Select other physiological channels to keep
# Adjust channel names as needed based on your EDF
physio_channels = ['LOC','ECG1','EMG1','THOR EFFORT', 'ABDO EFFORT', 'SaO2']
target_sfreq = 128  # Hz

for edf_path in sorted(input_dir.glob('*.edf')):
    print(f'Processing {edf_path.name}...')
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)

    # Extract C3 and M2 signals
    c3_signal = raw.copy().pick(['C3']).get_data()[0]
    m2_signal = raw.copy().pick(['M2']).get_data()[0]
    if len(c3_signal) != len(m2_signal):
        raise ValueError(f"C3 and M2 signals in {edf_path.name} do not have the same length")

    # Compute C3-M2 difference (fused channel)
    c3_m2_signal = (c3_signal - m2_signal).reshape(1, -1)
    sfreq = raw.info['sfreq']

    # Pick the available physio channels from original data
    physio_picks = [ch for ch in physio_channels if ch in raw.ch_names]
    if not physio_picks:
        print('  No physio channels found; skipping file.')
        continue

    physio_raw = raw.copy().pick(physio_picks)
    physio_data = physio_raw.get_data()

    # Combine fused C3-M2 channel with these physio signals
    combined_data = np.vstack([c3_m2_signal, physio_data])
    combined_info = mne.create_info(
        ch_names=['C3-M2'] + physio_picks,
        sfreq=sfreq,
        ch_types=['eeg'] + ['misc'] * len(physio_picks),
    )
    combined_raw = mne.io.RawArray(combined_data, combined_info)

    # Resample all channels to the target sampling frequency
    combined_raw.resample(target_sfreq, npad='auto')

    # Save combined data to new EDF file
    output_file = output_dir / f'{edf_path.stem}_combined.edf'
    combined_raw.export(str(output_file), fmt='edf')

    print(f'  Combined C3-M2 and physiological signals saved to {output_file}')