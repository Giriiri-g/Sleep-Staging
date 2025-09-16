import numpy as np

epochs = np.load(r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\processed\SC4001E0-PSG_epochs.npy")
labels = np.load(r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\processed\SC4001E0-PSG_labels.npy")

print("Epochs shape:", epochs.shape)  # (num_epochs, channels, 3000)
print("Labels shape:", labels.shape)  # (num_epochs,)
print("Unique labels:", np.unique(labels))
print("First 10 labels:", labels[:10])
