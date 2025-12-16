"""
Example script showing how to use the PSG preprocessing pipeline
"""

import os
from psg_preprocessing import PSGPreprocessor, preprocess_psg_file, preprocess_psg_folder
import mne

# Example 1: Preprocess a single file
def example_single_file():
    """Example of preprocessing a single PSG file"""
    print("=" * 60)
    print("Example 1: Preprocessing a single PSG file")
    print("=" * 60)
    
    # Path to your PSG file
    psg_file = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
    
    if not os.path.exists(psg_file):
        print(f"File not found: {psg_file}")
        print("Please update the path to your PSG file")
        return
    
    # Preprocess the file
    processed = preprocess_psg_file(
        file_path=psg_file,
        output_path=None,  # Will save as SC4001E0-PSG_preprocessed.edf
        target_sfreq=100.0,  # Resample to 100 Hz
        notch_freq=50.0,  # Remove 50 Hz line noise (use 60.0 for US)
        channel_specific_filter=True,  # Use channel-specific frequency bands
        artifact_method='threshold',  # Use threshold-based artifact removal
        save_preprocessed=True,
        verbose=True
    )
    
    print(f"\nPreprocessed file shape: {processed.get_data().shape}")
    print(f"Channels: {processed.ch_names}")
    print(f"Sampling frequency: {processed.info['sfreq']} Hz")


# Example 2: Preprocess entire folder
def example_folder():
    """Example of preprocessing all files in a folder"""
    print("=" * 60)
    print("Example 2: Preprocessing all PSG files in a folder")
    print("=" * 60)
    
    # Path to folder containing PSG files
    folder_path = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette"
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        print("Please update the path to your folder")
        return
    
    # Preprocess all files
    processed_files = preprocess_psg_folder(
        folder_path=folder_path,
        output_folder=None,  # Will save in same folder with '_preprocessed' suffix
        file_pattern="*-PSG.edf",  # Match all PSG files
        target_sfreq=100.0,
        notch_freq=50.0,
        channel_specific_filter=True,
        artifact_method='threshold',
        verbose=True
    )
    
    print(f"\nProcessed {len(processed_files)} files")


# Example 3: Custom preprocessing pipeline
def example_custom_pipeline():
    """Example of using the preprocessor class directly for custom pipeline"""
    print("=" * 60)
    print("Example 3: Custom preprocessing pipeline")
    print("=" * 60)
    
    # Path to your PSG file
    psg_file = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
    
    if not os.path.exists(psg_file):
        print(f"File not found: {psg_file}")
        print("Please update the path to your PSG file")
        return
    
    # Load raw data
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    print(f"Original: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels")
    
    # Create preprocessor
    preprocessor = PSGPreprocessor(
        target_sfreq=100.0,
        notch_freq=50.0,
        verbose=True
    )
    
    # Apply preprocessing steps selectively
    processed = raw.copy()
    
    # Step 1: Resample
    print("\nStep 1: Resampling...")
    processed = preprocessor.resample_raw(processed)
    
    # Step 2: Notch filter
    print("\nStep 2: Notch filter...")
    processed = preprocessor.apply_notch_filter(processed)
    
    # Step 3: Bandpass filter (channel-specific)
    print("\nStep 3: Bandpass filter...")
    processed = preprocessor.apply_bandpass_filter(
        processed,
        channel_specific=True
    )
    
    # Step 4: Detect bad channels
    print("\nStep 4: Detect bad channels...")
    bad_channels = preprocessor.detect_bad_channels(processed, method='auto')
    processed.info['bads'] = bad_channels
    
    # Step 5: Interpolate bad channels
    print("\nStep 5: Interpolate bad channels...")
    processed = preprocessor.interpolate_bad_channels(processed)
    
    # Step 6: Remove artifacts (threshold-based)
    print("\nStep 6: Remove artifacts...")
    processed = preprocessor.remove_artifacts_threshold(
        processed,
        threshold=5.0,
        method='zscore'
    )
    
    print(f"\nFinal: {processed.info['sfreq']} Hz, {len(processed.ch_names)} channels")
    print(f"Bad channels: {processed.info['bads']}")


# Example 4: Using ICA for artifact removal
def example_ica_artifacts():
    """Example of using ICA for artifact removal"""
    print("=" * 60)
    print("Example 4: ICA-based artifact removal")
    print("=" * 60)
    
    # Path to your PSG file
    psg_file = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
    
    if not os.path.exists(psg_file):
        print(f"File not found: {psg_file}")
        print("Please update the path to your PSG file")
        return
    
    # Preprocess with ICA
    processed = preprocess_psg_file(
        file_path=psg_file,
        target_sfreq=100.0,
        notch_freq=50.0,
        channel_specific_filter=True,
        artifact_method='ica',  # Use ICA for artifact removal
        save_preprocessed=False,
        verbose=True,
        # ICA-specific parameters
        ica_components=20,  # Number of ICA components
        ica_method='fastica',  # ICA method
        ecg_ch_name=None,  # ECG channel name (if available)
        eog_ch_name=None,  # EOG channel name (if available)
    )
    
    print(f"\nPreprocessed with ICA: {processed.get_data().shape}")


# Example 5: Custom frequency bands
def example_custom_frequencies():
    """Example of using custom frequency bands"""
    print("=" * 60)
    print("Example 5: Custom frequency bands")
    print("=" * 60)
    
    # Path to your PSG file
    psg_file = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
    
    if not os.path.exists(psg_file):
        print(f"File not found: {psg_file}")
        print("Please update the path to your PSG file")
        return
    
    # Load raw data
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    
    # Create preprocessor with custom frequency bands
    preprocessor = PSGPreprocessor(
        target_sfreq=100.0,
        notch_freq=50.0,
        verbose=True
    )
    
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
    
    # Preprocess
    processed = preprocessor.preprocess(
        raw,
        resample=True,
        notch_filter=True,
        bandpass_filter=True,
        channel_specific_filter=True,
        detect_bads=True,
        interpolate_bads=True,
        remove_artifacts=True,
        artifact_method='threshold'
    )
    
    print(f"\nPreprocessed with custom frequencies: {processed.get_data().shape}")


if __name__ == "__main__":
    # Run examples (comment out ones you don't want to run)
    
    # Example 1: Single file
    try:
        example_single_file()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: Folder (commented out to avoid processing all files)
    # try:
    #     example_folder()
    # except Exception as e:
    #     print(f"Example 2 failed: {e}")
    
    # Example 3: Custom pipeline
    try:
        example_custom_pipeline()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 4: ICA artifacts (commented out - takes longer)
    # try:
    #     example_ica_artifacts()
    # except Exception as e:
    #     print(f"Example 4 failed: {e}")
    
    # Example 5: Custom frequencies
    try:
        example_custom_frequencies()
    except Exception as e:
        print(f"Example 5 failed: {e}")










