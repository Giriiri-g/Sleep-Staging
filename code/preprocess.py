import os
import re
import numpy as np
import pandas as pd
import mne
from datetime import datetime, timedelta
import glob
from pathlib import Path

def parse_time_string(time_str):
    """Parse time string in HH:MM or HH:MM:SS format and return timedelta"""
    if pd.isna(time_str) or time_str == '':
        return None
    
    # Handle different time formats
    if isinstance(time_str, str):
        # Remove any extra spaces and handle formats like "22:44" or "22:44:30"
        time_str = time_str.strip()
        if ':' in time_str:
            time_parts = time_str.split(':')
            if len(time_parts) == 2:
                # HH:MM format
                hours, minutes = map(int, time_parts)
                return timedelta(hours=hours, minutes=minutes)
            elif len(time_parts) == 3:
                # HH:MM:SS format
                hours, minutes, seconds = map(int, time_parts)
                return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    return None

def extract_subject_night_from_filename(filename):
    """Extract subject ID and night number from PSG filename"""
    # Pattern: SC4ssNxx-PSG.edf where ss is 2-char subject, N is night, xx is clinician
    pattern = r'SC4(\d{2})(\d)([A-Z0-9]+)'
    match = re.search(pattern, filename)
    if match:
        subject_id = int(match.group(1))  # 2-digit subject ID
        night = int(match.group(2))       # single digit night number
        clinician_code = match.group(3)   # clinician initials
        
        return subject_id, night, clinician_code
    return None, None, None

def find_night_number_from_excel(df_subjects, subject_id, filename):
    """Try to determine night number by matching with Excel data"""
    # Get all rows for this subject
    subject_rows = df_subjects[df_subjects.iloc[:, 0] == subject_id]
    
    if len(subject_rows) == 1:
        # Only one night for this subject
        return int(subject_rows.iloc[0, 1])  # Return the night number from Excel
    elif len(subject_rows) == 2:
        # Two nights - we need to determine which one this file represents
        # This is tricky without more info, so we'll try both and see which works
        return None  # Will try both nights
    else:
        return None

def load_and_process_sleep_data(sleep_cassette_folder, excel_file_path, epoch_length=30.0):
    """
    Process Sleep-EDF Sleep Cassette dataset
    
    Parameters:
    -----------
    sleep_cassette_folder : str
        Path to folder containing Sleep-Cassette files
    excel_file_path : str
        Path to Excel file with subject information
    epoch_length : float
        Length of each epoch in seconds (default: 30.0)
    
    Returns:
    --------
    epochs_array : ndarray
        Array of PSG epochs
    labels_array : ndarray
        Array of corresponding sleep stage labels
    """
    
    # Load Excel file with subject information
    print("Loading Excel file...")
    try:
        df_subjects = pd.read_excel(excel_file_path)
        print(f"Loaded {len(df_subjects)} subject records from Excel file")
        print(f"Columns: {list(df_subjects.columns)}")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None, None
    
    # Initialize arrays to store epochs and labels
    epochs_list = []
    labels_list = []
    
    # Find all PSG files in the Sleep-Cassette folder
    psg_pattern = os.path.join(sleep_cassette_folder, "SC4*-PSG.edf")
    psg_files = glob.glob(psg_pattern)
    
    print(f"Found {len(psg_files)} PSG files")
    
    # Process each PSG file
    for psg_file in psg_files:
        try:
            filename = os.path.basename(psg_file)
            print(f"\nProcessing: {filename}")
            
            # Extract subject ID and night from filename
            result = extract_subject_night_from_filename(filename)
            if len(result) == 3:
                subject_id, night, clinician_code = result
                if subject_id is None:
                    print(f"Could not extract subject info from {filename}")
                    continue
            else:
                print(f"Could not extract subject info from {filename}")
                continue
            
            print(f"Subject ID: {subject_id}, Night: {night}, Clinician: {clinician_code}")
            
            # Find corresponding hypnogram file (clinician may be different)
            # Extract the base pattern without clinician ID
            base_pattern = f"SC4{subject_id:02d}{night}"
            hypnogram_pattern = os.path.join(sleep_cassette_folder, f"{base_pattern}*-Hypnogram.edf")
            hypnogram_files = glob.glob(hypnogram_pattern)
            
            if not hypnogram_files:
                print(f"No hypnogram file found for pattern: {base_pattern}*-Hypnogram.edf")
                continue
            elif len(hypnogram_files) > 1:
                print(f"Multiple hypnogram files found for {base_pattern}: {hypnogram_files}")
                print(f"Using first one: {hypnogram_files[0]}")
            
            hypnogram_file = hypnogram_files[0]
            print(f"Found hypnogram: {os.path.basename(hypnogram_file)}")
            
            # Find matching row in Excel file
            subject_row = df_subjects[(df_subjects.iloc[:, 0] == subject_id) & 
                                    (df_subjects.iloc[:, 1] == night)]
            
            if subject_row.empty:
                print(f"No matching row found for Subject {subject_id}, Night {night}")
                continue
            
            # Get LightsOff time from 5th column (index 4)
            lights_off_str = subject_row.iloc[0, 4]  # 5th column (0-indexed)
            lights_off_time = parse_time_string(str(lights_off_str))
            
            if lights_off_time is None:
                print(f"Could not parse LightsOff time: {lights_off_str}")
                continue
            
            print(f"LightsOff time: {lights_off_time}")
            
            # Load PSG data
            print("Loading PSG data...")
            raw_psg = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            print(f"PSG duration: {raw_psg.times[-1]:.2f} seconds ({raw_psg.times[-1]/3600:.2f} hours)")
            
            # Load hypnogram annotations (not as raw data)
            print("Loading hypnogram annotations...")
            try:
                # Try loading as annotations first (recommended for hypnogram files)
                annotations = mne.read_annotations(hypnogram_file)
                print(f"Loaded {len(annotations)} annotations")
                
                # Create a dummy raw object for hypnogram with same duration as PSG
                info = mne.create_info(['Hypnogram'], sfreq=1, ch_types=['misc'])
                hypno_data = np.zeros((1, int(raw_psg.times[-1])))  # Same duration as PSG
                raw_hypno = mne.io.RawArray(hypno_data, info)
                raw_hypno.set_annotations(annotations)
                
            except Exception as e:
                print(f"Loading as annotations failed: {e}")
                print("Loading hypnogram as EDF...")
                raw_hypno = mne.io.read_raw_edf(hypnogram_file, preload=True, verbose=False)
                print(f"Hypnogram duration: {raw_hypno.times[-1]:.2f} seconds")
            
            # Get recording start time
            recording_start = raw_psg.info['meas_date']
            if recording_start is None:
                print("Warning: No measurement date found in PSG file")
                # For Sleep Cassette study, recordings typically start in the evening
                # We'll assume the LightsOff time is relative to the start of recording
                crop_start_seconds = lights_off_time.total_seconds()
                print(f"Using LightsOff as relative time: {crop_start_seconds} seconds from recording start")
            else:
                print(f"Recording start time: {recording_start}")
                
                # Calculate the actual lights off time
                recording_start_time = recording_start.time()
                recording_start_seconds = (recording_start_time.hour * 3600 + 
                                        recording_start_time.minute * 60 + 
                                        recording_start_time.second)
                
                lights_off_seconds = lights_off_time.total_seconds()
                
                # Handle day rollover (if LightsOff is next day)
                if lights_off_seconds < recording_start_seconds:
                    lights_off_seconds += 24 * 3600  # Add 24 hours
                
                crop_start_seconds = lights_off_seconds - recording_start_seconds
                print(f"Recording starts at: {recording_start_seconds} seconds since midnight")
                print(f"LightsOff at: {lights_off_time.total_seconds()} seconds since midnight")
                print(f"Calculated crop start: {crop_start_seconds} seconds from recording start")
            
            # Validate crop time
            max_duration = min(raw_psg.times[-1], raw_hypno.times[-1] if hasattr(raw_hypno, 'times') else raw_psg.times[-1])
            
            print(f"Maximum available duration: {max_duration:.2f} seconds")
            print(f"Requested crop start: {crop_start_seconds:.2f} seconds")
            
            if crop_start_seconds >= max_duration:
                print(f"ERROR: Crop start ({crop_start_seconds:.2f}s) is beyond recording duration ({max_duration:.2f}s)")
                print("Skipping this file...")
                continue
            
            if crop_start_seconds < 0:
                print(f"WARNING: Negative crop start time ({crop_start_seconds:.2f}s), using 0")
                crop_start_seconds = 0
            
            # Crop PSG data from LightsOff time
            print(f"Cropping PSG data from {crop_start_seconds} seconds...")
            raw_psg_cropped = raw_psg.copy().crop(tmin=crop_start_seconds)
            
            # Crop hypnogram data from same time
            print("Cropping hypnogram data...")
            raw_hypno_cropped = raw_hypno.copy().crop(tmin=crop_start_seconds)
            
            # Create 30-second epochs for PSG
            print("Creating epochs...")
            epoch_duration = epoch_length
            n_epochs = int(raw_psg_cropped.times[-1] // epoch_duration)
            
            # Get PSG data
            psg_data = raw_psg_cropped.get_data()  # Shape: (n_channels, n_samples)
            
            # Get hypnogram annotations
            hypno_data = raw_hypno_cropped.get_data()
            
            # Sampling frequency
            sfreq = raw_psg_cropped.info['sfreq']
            samples_per_epoch = int(epoch_duration * sfreq)
            
            print(f"Creating {n_epochs} epochs of {epoch_duration} seconds each")
            
            # Extract epochs and corresponding labels
            for epoch_idx in range(n_epochs):
                start_sample = epoch_idx * samples_per_epoch
                end_sample = start_sample + samples_per_epoch
                
                if end_sample > psg_data.shape[1]:
                    break
                
                # Extract PSG epoch
                psg_epoch = psg_data[:, start_sample:end_sample]
                
                # Extract corresponding sleep stage from hypnogram
                # Sample at the middle of the epoch for label
                label_sample = start_sample + samples_per_epoch // 2
                if label_sample < hypno_data.shape[1]:
                    # Get the most common annotation in this epoch
                    hypno_epoch = hypno_data[:, start_sample:end_sample]
                    # For simplicity, take the value at the middle of the epoch
                    sleep_stage = hypno_data[0, label_sample]
                    
                    epochs_list.append(psg_epoch)
                    labels_list.append(sleep_stage)
            
            print(f"Added {n_epochs} epochs from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if not epochs_list:
        print("No epochs were successfully processed!")
        return None, None
    
    # Convert lists to numpy arrays
    print("\nConverting to numpy arrays...")
    epochs_array = np.array(epochs_list)
    labels_array = np.array(labels_list)
    
    print(f"Final epochs array shape: {epochs_array.shape}")
    print(f"Final labels array shape: {labels_array.shape}")
    print(f"Unique labels: {np.unique(labels_array)}")
    
    return epochs_array, labels_array

def main():
    """Main function to process the dataset"""
    
    # Configuration
    sleep_cassette_folder = "Sleep-Cassette"  # Update this path
    excel_file_path = "SC-subjects.xls"  # Update this path
    epoch_length = 30.0  # 30 seconds
    
    print("Starting Sleep-EDF dataset processing...")
    print(f"Sleep Cassette folder: {sleep_cassette_folder}")
    print(f"Excel file: {excel_file_path}")
    
    # Check if paths exist
    if not os.path.exists(sleep_cassette_folder):
        print(f"Error: Sleep-Cassette folder not found: {sleep_cassette_folder}")
        return
    
    if not os.path.exists(excel_file_path):
        print(f"Error: Excel file not found: {excel_file_path}")
        return
    
    # Process the data
    epochs_array, labels_array = load_and_process_sleep_data(
        sleep_cassette_folder, excel_file_path, epoch_length
    )
    
    if epochs_array is not None and labels_array is not None:
        # Save the processed data
        print("\nSaving processed data...")
        
        np.save("preprocessed_Epochs.npy", epochs_array)
        np.save("preprocessed_Labels.npy", labels_array)
        
        print("Saved preprocessed_Epochs.npy")
        print("Saved preprocessed_Labels.npy")
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total epochs: {len(epochs_array)}")
        print(f"Epoch shape: {epochs_array[0].shape}")
        print(f"Channels: {epochs_array.shape[1]}")
        print(f"Samples per epoch: {epochs_array.shape[2]}")
        
        # Count sleep stages
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        print("\nSleep stage distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Stage {label}: {count} epochs")
        
        print("\nProcessing completed successfully!")
    else:
        print("Processing failed - no data was saved.")

if __name__ == "__main__":
    main()