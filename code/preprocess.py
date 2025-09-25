import mne
import numpy as np
import os
import glob

# Your folder path
folder = r'C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-telemetry'
output_folder = os.path.join(folder, "processed")
os.makedirs(output_folder, exist_ok=True)

# List all PSG and hypnogram files
psg_files = sorted(glob.glob(os.path.join(folder, '*-PSG.edf')))
hypnogram_files = sorted(glob.glob(os.path.join(folder, '*-Hypnogram.edf')))

# Build a lookup: patient_id -> hypnogram file (using first 6 chars as patient ID)
hypno_dict = {}
for file in hypnogram_files:
    base = os.path.basename(file)
    patient_id = base[:6]  # First 6 characters (SC4XXX)
    hypno_dict[patient_id] = file

print(f"Found {len(psg_files)} PSG files and {len(hypnogram_files)} hypnogram files")
print(f"Built hypnogram dictionary with {len(hypno_dict)} entries")

def preprocess_file(psg_path, hypno_path, output_folder):
    print(f"Processing: {os.path.basename(psg_path)} + {os.path.basename(hypno_path)}")
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    annot = mne.read_annotations(hypno_path)
    raw.set_annotations(annot, emit_warning=False)
    raw.filter(0.5, 32.0)
    raw.resample(100)

    # Extract events from hypnogram
    events, event_id = mne.events_from_annotations(raw)
    epoch_length = 30
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=0, tmax=epoch_length-1/raw.info['sfreq'],
                        baseline=None, preload=True)

    # Epoch data
    clean_epochs = []
    labels = []
    def reject_criteria(data): return np.any(np.abs(data) > 1000)

    for i, ep in enumerate(epochs.get_data()):
        if not reject_criteria(ep):
            # normalize
            ep = (ep - ep.mean(axis=1, keepdims=True)) / (ep.std(axis=1, keepdims=True) + 1e-6)
            clean_epochs.append(ep)
            labels.append(epochs.events[i, -1])  # sleep stage label

    clean_epochs = np.array(clean_epochs)
    labels = np.array(labels)

    base = os.path.splitext(os.path.basename(psg_path))[0]
    np.save(os.path.join(output_folder, f'{base}_epochs.npy'), clean_epochs)
    np.save(os.path.join(output_folder, f'{base}_labels.npy'), labels)

    print(f"Processed and saved: {base}, epochs={clean_epochs.shape}, labels={labels.shape}")


# Loop and pair files by patient ID (first 6 characters)
matched = 0
for psg_path in psg_files:
    base = os.path.basename(psg_path)
    patient_id = base[:6]  # First 6 characters (SC4XXX)
    hypno_path = hypno_dict.get(patient_id)
    if hypno_path:
        preprocess_file(psg_path, hypno_path, output_folder)
        matched += 1
    else:
        print(f"No hypnogram found for {base}")

print(f"Successfully matched and processed {matched} patient files")
