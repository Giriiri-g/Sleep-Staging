import os
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Parameters
edf_folder = 'path_to_edf_files'  # Folder with EDF files to process
output_folder = 'embeddings_output'  # Folder to save embeddings
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

window_duration_sec = 30  # Window length for epoching
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}
channels_of_interest = None  # Use None to select all channels in EDF, or set list e.g. ['C3', 'M2', 'EOG1', ...]

def extract_psd_embeddings(raw, sfreq, freq_bands):
    # Compute PSD with Welch
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=30, n_fft=2048)
    
    # Extract mean power in each band per channel
    features = []
    for band, (fmin, fmax) in freq_bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(psd[:, idx_band], axis=1)
        features.append(band_power)
    
    # Concatenate features (channels x bands)
    features = np.concatenate(features)
    return features

def process_edf_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Select channels if specified
    if channels_of_interest:
        picks = [ch for ch in channels_of_interest if ch in raw.ch_names]
        raw.pick_channels(picks)
    
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    window_samples = int(window_duration_sec * sfreq)
    n_windows = n_samples // window_samples
    
    embeddings_list = []
    for win_idx in range(n_windows):
        start = win_idx * window_samples
        stop = start + window_samples
        epoch = raw.copy().crop(tmin=start/sfreq, tmax=(stop-1)/sfreq)
        embedding = extract_psd_embeddings(epoch, sfreq, freq_bands)
        embeddings_list.append(embedding)
    
    embeddings_array = np.vstack(embeddings_list)  # Shape: (n_windows, n_channels*n_bands)
    return embeddings_array

# Process all EDF files in the folder
for edf_file in os.listdir(edf_folder):
    if edf_file.lower().endswith('.edf'):
        full_path = os.path.join(edf_folder, edf_file)
        print(f'Processing {edf_file}...')
        embeddings = process_edf_file(full_path)
        
        # Save embeddings as CSV
        output_csv = os.path.join(output_folder, edf_file.replace('.edf', '_embeddings.csv'))
        pd.DataFrame(embeddings).to_csv(output_csv, index=False)
        print(f'Saved embeddings to {output_csv}')
