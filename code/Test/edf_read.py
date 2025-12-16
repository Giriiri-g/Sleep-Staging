import mne

def print_edf_channel_names(edf_file_path):
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=False)
    
    # Get channel names
    channel_names = raw.info['ch_names']
    
    # Print each channel name
    print("Channels in EDF file:")
    for ch in channel_names:
        print(ch)

# Example usage
edf_path = r'D:\MESA_Sample\edf\mesa-sleep-0001.edf'

print_edf_channel_names(edf_path)
