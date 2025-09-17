import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import spectrogram

# Paths to processed folders
folders = [
    "sleep-edf-database-expanded-1.0.0/sleep-cassette/processed",
    "sleep-edf-database-expanded-1.0.0/sleep-telemetry/processed"
]

epochs_all, labels_all, subjects, ages = [], [], [], []

def parse_subject_info(filename):
    """
    Extracts subject ID (first 6 characters of filename).
    Example: SC4001E0-PSG_epochs.npy -> SC4001
    """
    return filename[:6]

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith("_epochs.npy"):
            epoch_file = os.path.join(folder, file)
            label_file = epoch_file.replace("_epochs.npy", "_labels.npy")

            if not os.path.exists(label_file):
                continue

            epochs = np.load(epoch_file)
            labels = np.load(label_file)

            subj_id = parse_subject_info(file)

            epochs_all.append(epochs)
            labels_all.append(labels)
            subjects.extend([subj_id] * len(labels))

            # Placeholder for subject age (replace with actual metadata later)
            try:
                subj_num = int(subj_id[2:])  # e.g., 4001 -> 4001
            except:
                subj_num = -1
            subj_age = 20 + (subj_num % 70)  # crude synthetic age mapping
            ages.extend([subj_age] * len(labels))

# Concatenate
X = np.concatenate(epochs_all, axis=0)
y = np.concatenate(labels_all, axis=0)
subjects = np.array(subjects)
ages = np.array(ages)

print("Final dataset:", X.shape, y.shape)
print("Unique labels:", np.unique(y))

# --- EDA ---

# Label distribution
plt.figure(figsize=(8,5))
sns.countplot(x=y)
plt.title("Label Distribution (All Subjects)")
plt.show()

# Label distribution for elderly
plt.figure(figsize=(8,5))
sns.countplot(x=y[ages > 60])
plt.title("Label Distribution (Age > 60)")
plt.show()

# Channel variance
channel_var = X.var(axis=(0,2))  # variance per channel
plt.figure(figsize=(10,6))
sns.barplot(x=list(range(X.shape[1])), y=channel_var)
plt.title("Channel Variance Across Dataset")
plt.xlabel("Channel")
plt.ylabel("Variance")
plt.show()

# Example spectrogram for elderly subject
rand_idx = np.where(ages > 60)[0][0]
f, t, Sxx = spectrogram(X[rand_idx, 0, :], fs=100)  # adjust fs if needed
plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title("Spectrogram (Age > 60 sample)")
plt.colorbar(label='Power [dB]')
plt.show()

# Summary statistics
df = pd.DataFrame({
    "Subject": subjects,
    "Age": ages,
    "Label": y
})

print("\n--- Summary ---")
print(df.groupby("Age")["Label"].count().describe())
print("\nLabel counts by age group:")
print(df.groupby(pd.cut(df["Age"], bins=[0,40,60,80,100]))["Label"].count())
