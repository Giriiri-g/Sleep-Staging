from pathlib import Path

import mne

# Configure your input and output directories here
input_dir = Path('path/to/edf_directory')
output_dir = Path('path/to/output_directory')
output_dir.mkdir(parents=True, exist_ok=True)

# Select other physiological channels to keep
# Adjust channel names as needed based on your EDF
physio_channels = ['EEG(sec)','EOG(L)','ECG','EMG','THOR RES', 'ABDO RES', 'SaO2']
target_sfreq = 128  # Hz

for edf_path in sorted(input_dir.glob('*.edf')):
    print(f'Processing {edf_path.name}...')
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)

    # Pick the available physio channels from original data
    physio_picks = [ch for ch in physio_channels if ch in raw.ch_names]
    physio_raw = raw.copy().pick_channels(physio_picks)

    # Resample all channels to the target sampling frequency
    physio_raw.resample(target_sfreq, npad='auto')

    # Save combined data to new EDF file
    output_file = output_dir / f'{edf_path.stem}_combined.edf'
    physio_raw.export(str(output_file), fmt='edf')

    print(f'  Combined C3-M2 and physiological signals saved to {output_file}')