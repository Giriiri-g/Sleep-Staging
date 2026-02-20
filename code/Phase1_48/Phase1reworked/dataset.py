import torch
import pandas as pd
from torch.utils.data import Dataset

STAGE_TO_IDX = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4
}

IDX_TO_STAGE = {v: k for k, v in STAGE_TO_IDX.items()}
NUM_CLASSES = 5


class SleepEDFSequenceDataset(Dataset):
    """
    Lazy-loading, sliding-window dataset.

    Each sample:
        x: [T, 3000]   where T <= window_size
        y: [T]
    """

    def __init__(
        self,
        csv_path,
        feature='temporal',
        file_indices=None,
        window_size=256,
        overlap=64
    ):
        assert overlap < window_size, "overlap must be smaller than window_size"

        self.df = pd.read_csv(csv_path)
        if file_indices is not None:
            self.df = self.df.iloc[file_indices].reset_index(drop=True)

        self.window_size = window_size
        self.stride = window_size - overlap
        self.feature = feature

        if feature == 'temporal':
            self.path = 'tensor_path'
        elif feature == 'spectral':
            self.path = 'spectral'
        elif feature == 'fusion':
            self.temporal_path = 'tensor_path'
            self.spectral_path = 'spectral'
        else:
            raise ValueError("feature must be temporal | spectral | fusion")
        # Build index: (row_idx, start_epoch)
        self.index = []

        for row_idx, row in self.df.iterrows():
            stages = row["stage_sequence"].split(" ")
            T = len(stages)

            start = 0
            while start < T:
                self.index.append((row_idx, start))
                start += self.stride

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row_idx, start = self.index[idx]
        row = self.df.iloc[row_idx]

        end = start + self.window_size
        stages = row["stage_sequence"].split(" ")

        y = torch.tensor(
            [STAGE_TO_IDX[s] for s in stages[start:end]],
            dtype=torch.long
        )

        if self.feature == 'fusion':
            x_temporal = torch.load(row[self.temporal_path])
            x_spectral = torch.load(row[self.spectral_path])

            x_temporal = x_temporal[start:end]
            x_spectral = x_spectral[start:end]

            return (x_temporal.float(), x_spectral.float()), y

        else:
            x = torch.load(row[self.path])
            x = x[start:end]
            return x.float(), y
