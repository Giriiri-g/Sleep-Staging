import mne
import numpy as np
import os
import glob
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
folder = r'sleep-edf-database-expanded-1.0.0\sleep-cassette'
meta_path = r'sleep-edf-database-expanded-1.0.0\SC-subjects.xls'
output_folder = os.path.join(folder, "processed")
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Load metadata
# ----------------------------
meta_df = pd.read_excel(meta_path)
print("Metadata preview:")
print(meta_df.head())

# Build lookup: (subject, night) -> (age, sex, lights_off_seconds)
meta_lookup = {}
for _, row in meta_df.iterrows():
    subj = int(row['subject'])
    night = int(row['night'])
    age = int(row['age'])
    sex = int(row['sex (F=1)'])  # 1 = female, 0 = male
    # convert HH:MM:SS -> seconds
    h, m, s = map(int, str(row['LightsOff']).split(':'))
    lights_off_sec = h * 3600 + m * 60 + s
    meta_lookup[(subj, night)] = (age, sex, lights_off_sec)

# ----------------------------
# List PSG and Hypnogram files
# ----------------------------
psg_files = sorted(glob.glob(os.path.join(folder, '*-PSG.edf')))
hypnogram_files = sorted(glob.glob(os.path.join(folder, '*-Hypnogram.edf')))

# Build hypnogram lookup
hypno_dict = {}
for file in hypnogram_files:
    base = os.path.basename(file)
    patient_id = base[:6]  # First 6 characters (SC4XXX)
    hypno_dict[patient_id] = file

print(f"Found {len(psg_files)} PSG files and {len(hypnogram_files)} hypnogram files")

# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess_file(psg_path, hypno_path, output_folder):
    base = os.path.splitext(os.path.basename(psg_path))[0]
    patient_id = base[:6]  # e.g. SC4XXX
    subj = int(patient_id[2:4])  # SC4XX → subject number
    night = int(patient_id[4])   # last digit = night (1/2)

    # Metadata
    if (subj, night) not in meta_lookup:
        print(f"⚠️ No metadata found for {patient_id}")
        return 0
    age, sex, lights_off_sec = meta_lookup[(subj, night)]

    print(f"Processing {patient_id} (Age={age}, Sex={sex}, LightsOff={lights_off_sec}s)")

    # Load PSG and Hypnogram
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hypno_path)
    raw.set_annotations(annot, emit_warning=False)

    # Crop from LightsOff
    raw.crop(tmin=lights_off_sec)

    # Filtering and resampling
    raw.filter(0.5, 32.0, verbose=False)
    raw.resample(100, verbose=False)

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epoch_length = 30
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=0, tmax=epoch_length-1/raw.info['sfreq'],
                        baseline=None, preload=True, verbose=False)

    # Epoch data
    clean_epochs = []
    labels = []

    def reject_criteria(data): return np.any(np.abs(data) > 1000)

    for i, ep in enumerate(epochs.get_data()):
        if not reject_criteria(ep):
            # normalize per channel
            ep = (ep - ep.mean(axis=1, keepdims=True)) / (ep.std(axis=1, keepdims=True) + 1e-6)
            # add age + sex (as constant channels)
            age_sex = np.array([age, sex]).reshape(2, 1)
            ep_aug = np.vstack([ep, age_sex])  # (channels+2, time)
            clean_epochs.append(ep_aug)
            labels.append(epochs.events[i, -1])  # sleep stage label

    clean_epochs = np.array(clean_epochs)
    labels = np.array(labels)

    # Save
    np.save(os.path.join(output_folder, f'{base}_epochs.npy'), clean_epochs)
    np.save(os.path.join(output_folder, f'{base}_labels.npy'), labels)

    print(f"✔ {base} → epochs={clean_epochs.shape}, labels={labels.shape}")
    return clean_epochs.shape[0]

# ----------------------------
# Run preprocessing
# ----------------------------
total_epochs = 0
matched = 0
for psg_path in psg_files:
    base = os.path.basename(psg_path)
    patient_id = base[:6]
    hypno_path = hypno_dict.get(patient_id)
    if hypno_path:
        n_epochs = preprocess_file(psg_path, hypno_path, output_folder)
        total_epochs += n_epochs
        matched += 1
    else:
        print(f"No hypnogram found for {base}")

print(f"\n✅ Successfully processed {matched} patients")
print(f"📊 Total number of epochs after segmentation: {total_epochs}")
