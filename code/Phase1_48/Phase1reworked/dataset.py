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
    Each sample:
      x: [2W+1, 3000]
      y: scalar (center epoch label)
    """

    def __init__(self, csv_path, window=5):
        self.df = pd.read_csv(csv_path)
        self.window = window
        self.samples = []

        for _, row in self.df.iterrows():
            x = torch.load(row["tensor_path"])  # [T, 3000]
            stages = row["stage_sequence"].split(" ")
            y = torch.tensor([STAGE_TO_IDX[s] for s in stages])

            T = len(y)
            for t in range(T):
                self.samples.append((x, y, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, t = self.samples[idx]
        W = self.window
        L = x.shape[1]

        start = max(0, t - W)
        end = min(len(y), t + W + 1)

        window_x = x[start:end]

        # left padding
        if start > t - W:
            pad = torch.zeros((W - (t - start), L))
            window_x = torch.cat([pad, window_x], dim=0)

        # right padding
        if end < t + W + 1:
            pad = torch.zeros((t + W + 1 - end, L))
            window_x = torch.cat([window_x, pad], dim=0)

        return window_x.float(), y[t]
