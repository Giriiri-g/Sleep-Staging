import matplotlib
matplotlib.use('TkAgg')  # Ensure GUI backend is set; try 'Qt5Agg' if TkAgg has issues

import mne

# Path to PSG and annotation (hypnogram) files
psg_path = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
hyp_path = r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001EC-Hypnogram.edf"

# Load raw PSG
raw = mne.io.read_raw_edf(psg_path, preload=True)

# Load annotation/hypnogram if available
try:
    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations)
except Exception as e:
    print(f"Annotation loading failed: {e}")

print("Channels:", raw.ch_names)
print("Bad channels:", raw.info['bads'])

# Visualize a 60-second segment starting at 0 seconds
raw.plot(start=0, duration=60, block=True)

input("Press Enter to exit...")  # Ensures the window stays open if running as a script
