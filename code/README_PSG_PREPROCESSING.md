# PSG Preprocessing Script

Comprehensive preprocessing pipeline for Polysomnography (PSG) datasets with support for channel resampling, artifact removal, bandpass filtering, and interpolation.

## Features

- **Channel Resampling**: Resample all channels to a target sampling frequency (default: 100 Hz)
- **Artifact Removal**: Multiple methods including threshold-based and ICA-based removal
- **Bandpass Filtering**: Channel-specific frequency bands for different signal types (EEG, EOG, EMG, etc.)
- **Interpolation**: Automatic detection and interpolation of bad channels
- **Notch Filtering**: Remove line noise (50 Hz or 60 Hz)
- **Flexible Pipeline**: Apply all steps or select specific preprocessing steps

## Installation

Make sure you have the required dependencies:

```bash
pip install mne scipy numpy
```

## Quick Start

### Preprocess a Single File

```python
from psg_preprocessing import preprocess_psg_file

# Preprocess a single PSG file
processed = preprocess_psg_file(
    file_path="path/to/your/file.edf",
    target_sfreq=100.0,  # Resample to 100 Hz
    notch_freq=50.0,  # Remove 50 Hz line noise (use 60.0 for US)
    channel_specific_filter=True,  # Use channel-specific frequency bands
    artifact_method='threshold',  # Artifact removal method
    save_preprocessed=True,
    verbose=True
)
```

### Preprocess All Files in a Folder

```python
from psg_preprocessing import preprocess_psg_folder

# Preprocess all PSG files in a folder
processed_files = preprocess_psg_folder(
    folder_path="path/to/folder",
    target_sfreq=100.0,
    notch_freq=50.0,
    channel_specific_filter=True,
    artifact_method='threshold',
    verbose=True
)
```

## Command Line Usage

### Single File

```bash
python psg_preprocessing.py --input path/to/file.edf --output path/to/output.edf --target_sfreq 100 --notch_freq 50 --channel_specific --verbose
```

### Entire Folder

```bash
python psg_preprocessing.py --input path/to/folder --output path/to/output_folder --folder --pattern "*-PSG.edf" --target_sfreq 100 --notch_freq 50 --channel_specific --verbose
```

## Preprocessing Steps

### 1. Resampling

Resamples all channels to a target sampling frequency (default: 100 Hz).

```python
preprocessor = PSGPreprocessor(target_sfreq=100.0)
processed = preprocessor.resample_raw(raw)
```

### 2. Notch Filtering

Removes line noise (50 Hz or 60 Hz) using a notch filter.

```python
processed = preprocessor.apply_notch_filter(raw, freqs=50.0)
```

### 3. Bandpass Filtering

Applies bandpass filtering with channel-specific frequency bands:

- **EEG**: 0.5-35 Hz
- **EOG**: 0.1-15 Hz
- **EMG**: 10-100 Hz
- **ECG**: 0.5-40 Hz
- **Respiratory**: 0.1-5 Hz
- **Temperature**: 0.01-0.5 Hz

```python
# Channel-specific filtering (recommended)
processed = preprocessor.apply_bandpass_filter(raw, channel_specific=True)

# Or use same filter for all channels
processed = preprocessor.apply_bandpass_filter(raw, channel_specific=False, low_freq=0.5, high_freq=35.0)
```

### 4. Bad Channel Detection

Automatically detects bad channels using multiple criteria:

- **Flat channels**: Channels with very low variance
- **Noisy channels**: Channels with very high variance
- **Low correlation**: Channels with low correlation to other channels

```python
bad_channels = preprocessor.detect_bad_channels(raw, method='auto')
```

### 5. Interpolation

Interpolates bad channels using surrounding good channels.

```python
# Using MNE's interpolation (recommended for EEG/EOG)
processed = preprocessor.interpolate_bad_channels(raw, method='MNE')

# Or using scipy interpolation (fallback)
processed = preprocessor.interpolate_bad_channels(raw, method='scipy')
```

### 6. Artifact Removal

Two methods available:

#### Threshold-Based (Fast)

Removes artifacts by clipping values beyond a threshold (default: 5 standard deviations).

```python
processed = preprocessor.remove_artifacts_threshold(
    raw,
    threshold=5.0,
    method='zscore'  # or 'mad' for median absolute deviation
)
```

#### ICA-Based (Accurate but Slower)

Uses Independent Component Analysis to detect and remove artifacts.

```python
processed = preprocessor.remove_artifacts_ica(
    raw,
    n_components=20,
    method='fastica',  # or 'infomax', 'picard'
    ecg_ch_name='ECG',  # Optional: ECG channel name
    eog_ch_name='EOG'   # Optional: EOG channel name
)
```

## Complete Pipeline

Use the complete preprocessing pipeline:

```python
from psg_preprocessing import PSGPreprocessor

# Create preprocessor
preprocessor = PSGPreprocessor(
    target_sfreq=100.0,
    notch_freq=50.0,
    verbose=True
)

# Apply complete pipeline
processed = preprocessor.preprocess(
    raw,
    resample=True,
    notch_filter=True,
    bandpass_filter=True,
    channel_specific_filter=True,
    detect_bads=True,
    interpolate_bads=True,
    remove_artifacts=True,
    artifact_method='threshold'  # or 'ica', 'both'
)
```

## Custom Frequency Bands

You can customize frequency bands for different signal types:

```python
preprocessor = PSGPreprocessor(target_sfreq=100.0)

# Customize frequency bands
preprocessor.freq_bands = {
    'eeg': (0.5, 40.0),      # Wider EEG band
    'eog': (0.1, 20.0),      # Wider EOG band
    'emg': (10.0, 120.0),    # Wider EMG band
    'ecg': (0.5, 50.0),      # Wider ECG band
    'resp': (0.1, 10.0),     # Wider respiratory band
    'temp': (0.01, 1.0),     # Wider temperature band
    'default': (0.5, 40.0)   # Default band
}

# Apply preprocessing
processed = preprocessor.preprocess(raw)
```

## Parameters

### PSGPreprocessor

- `target_sfreq` (float): Target sampling frequency (default: 100.0 Hz)
- `notch_freq` (float or None): Notch filter frequency for line noise (50 or 60 Hz, None to disable)
- `notch_quality` (float): Quality factor for notch filter (default: 30.0)
- `filter_method` (str): Filter method ('fir' or 'iir', default: 'fir')
- `filter_length` (str): Filter length ('auto', '10s', etc., default: 'auto')
- `verbose` (bool): Whether to print verbose output (default: False)

### preprocess() method

- `resample` (bool): Whether to resample to target frequency (default: True)
- `notch_filter` (bool): Whether to apply notch filter (default: True)
- `bandpass_filter` (bool): Whether to apply bandpass filter (default: True)
- `channel_specific_filter` (bool): Whether to use channel-specific frequency bands (default: True)
- `detect_bads` (bool): Whether to detect bad channels (default: True)
- `interpolate_bads` (bool): Whether to interpolate bad channels (default: True)
- `remove_artifacts` (bool): Whether to remove artifacts (default: True)
- `artifact_method` (str): Artifact removal method ('threshold', 'ica', or 'both', default: 'threshold')

### Additional Parameters (kwargs)

- `low_freq` (float): Low cutoff frequency for bandpass filter
- `high_freq` (float): High cutoff frequency for bandpass filter
- `bad_detection_method` (str): Bad channel detection method ('auto', 'flat', 'noisy', 'correlation', 'all')
- `flat_criteria` (dict): Criteria for flat channel detection
- `noisy_criteria` (dict): Criteria for noisy channel detection
- `interpolation_method` (str): Interpolation method ('MNE' or 'scipy')
- `artifact_threshold` (float): Threshold for artifact detection (default: 5.0)
- `threshold_method` (str): Threshold method ('zscore' or 'mad')
- `ica_components` (int): Number of ICA components
- `ica_method` (str): ICA method ('fastica', 'infomax', 'picard')
- `ecg_ch_name` (str): ECG channel name for ICA
- `eog_ch_name` (str): EOG channel name for ICA
- `reject` (dict): Rejection parameters for epoching

## Examples

See `example_psg_preprocessing.py` for detailed examples:

1. **Single file preprocessing**
2. **Folder preprocessing**
3. **Custom pipeline**
4. **ICA-based artifact removal**
5. **Custom frequency bands**

## Signal Type Detection

The preprocessor automatically detects signal types from channel names:

- **EEG**: Channels containing 'eeg' in the name
- **EOG**: Channels containing 'eog' in the name
- **EMG**: Channels containing 'emg' in the name
- **ECG**: Channels containing 'ecg' or 'ekg' in the name
- **Respiratory**: Channels containing 'resp' or 'breath' in the name
- **Temperature**: Channels containing 'temp' or 'temperature' in the name
- **Default**: All other channels use the default frequency band

## Notes

1. **Sampling Frequency**: The default target sampling frequency is 100 Hz, which is standard for sleep staging. Adjust based on your needs.

2. **Line Noise**: Use 50 Hz for European datasets and 60 Hz for US datasets. Set to `None` to disable notch filtering.

3. **Artifact Removal**: 
   - **Threshold-based**: Fast and suitable for most cases
   - **ICA-based**: More accurate but slower, recommended for high-quality data
   - **Both**: Apply threshold first, then ICA (most thorough but slowest)

4. **Memory Usage**: For very long recordings (>1 hour), ICA automatically uses a subset of data for fitting to save memory.

5. **Channel Selection**: The preprocessor works with any number of channels. It automatically detects signal types and applies appropriate filters.

## Integration with Existing Code

The preprocessor is compatible with MNE-Python and can be used with existing MNE workflows:

```python
from psg_preprocessing import PSGPreprocessor
import mne

# Load raw data
raw = mne.io.read_raw_edf("file.edf", preload=True)

# Preprocess
preprocessor = PSGPreprocessor(target_sfreq=100.0)
processed = preprocessor.preprocess(raw)

# Continue with MNE operations
epochs = mne.make_fixed_length_epochs(processed, duration=30.0)
# ... rest of your code
```

## Troubleshooting

### Memory Issues

If you encounter memory issues with large files:

1. Use threshold-based artifact removal instead of ICA
2. Process files individually instead of in batch
3. Reduce the number of ICA components

### Channel Detection Issues

If channels are not detected correctly:

1. Check channel names in the EDF file
2. Manually specify frequency bands using `freq_bands` parameter
3. Use `channel_specific_filter=False` to apply same filter to all channels

### Interpolation Issues

If interpolation fails:

1. Check that there are enough good channels (at least 2)
2. Try using `method='scipy'` instead of `method='MNE'`
3. Manually mark bad channels before interpolation

## Citation

If you use this preprocessing script in your research, please cite:

- MNE-Python: https://mne.tools/
- Scipy: https://scipy.org/

## License

This script is provided as-is for research purposes.







