import numpy as np

# Load the .npy file
data = np.load("sleep-edf-database-expanded-1.0.0/preprocessed_Epochs.npy")

# Number of dimensions and shape
print("Shape:", data.shape)

# Total number of elements
print("Number of entries:", data.size)



# Load labels file
labels = np.load("sleep-edf-database-expanded-1.0.0/preprocessed_Labels.npy")

# Check shape and type
print("Shape:", labels.shape)
print("Data type:", labels.dtype)

# Count unique labels
unique_labels, counts = np.unique(labels, return_counts=True)

print("Unique labels:", unique_labels)
print("Counts:", counts)

# Make it more readable
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")
