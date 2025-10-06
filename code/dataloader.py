import os
import re
import mne
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LazyPSGDataset(Dataset):
    def __init__(self, folder_path, window_size=30):
        self.folder_path = folder_path
        self.window_size = window_size

        self.stage_mapping = {
            'Sleep stage W': 0,
            'Sleep stage R': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3': 4,
            'Sleep stage 4': 5,
            'Sleep stage ?': 6  # unscored
        }

        # List all PSG files and prepare epoch index mapping
        self.index_map = []  # List of tuples (psg_file, onset_sample, label)
        self.sfreq_cache = {}  # Cache sampling frequencies for files

        self._prepare_index()

    def _find_annotation_file(self, psg_filename):
        base = psg_filename[:6]
        pattern = re.compile(rf'{base}..-Hypnogram.edf')
        for f in os.listdir(self.folder_path):
            if pattern.fullmatch(f):
                return os.path.join(self.folder_path, f)
        return None

    def _prepare_index(self):
        for psg_file in os.listdir(self.folder_path):
            if psg_file.endswith('-PSG.edf'):
                psg_path = os.path.join(self.folder_path, psg_file)
                hyp_path = self._find_annotation_file(psg_file)
                if not hyp_path:
                    print(f"No annotation file for {psg_file}, skipping.")
                    continue

                raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
                sfreq = int(raw.info['sfreq'])
                self.sfreq_cache[psg_file] = sfreq

                annotations = mne.read_annotations(hyp_path)

                for desc, onset, duration in zip(annotations.description, annotations.onset, annotations.duration):
                    label = self.stage_mapping.get(desc, 6)
                    full_epochs = int(duration // self.window_size)
                    epoch_samples = self.window_size * sfreq
                    start_sample = int(onset * sfreq)

                    for i in range(full_epochs):
                        epoch_start = start_sample + i * epoch_samples
                        # Store tuple for lazy loading
                        self.index_map.append((psg_file, epoch_start, label))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        psg_file, start_sample, label = self.index_map[idx]
        psg_path = os.path.join(self.folder_path, psg_file)
        sfreq = self.sfreq_cache[psg_file]
        epoch_samples = self.window_size * sfreq

        # Load raw data for this file but only pick needed samples
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)

        data, times = raw[:, start_sample:start_sample+epoch_samples]
        data_tensor = torch.tensor(data, dtype=torch.float32)

        return data_tensor, label


if __name__ == '__main__':
    folder = r'C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette'
    dataset = LazyPSGDataset(folder_path=folder)
    print(f"Total epochs in dataset: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    for i, (batch_x, batch_y) in enumerate(loader):
        print(f"Batch {i} - Data shape: {batch_x.shape}, Labels: {batch_y}")
        if i == 3:  # limit to first 4 batches
            break

