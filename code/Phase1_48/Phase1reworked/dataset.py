import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = torch.load(row["tensor_path"])        # [T, 3000]
        stages = row["stage_sequence"].split(" ")
        y = torch.tensor([STAGE_TO_IDX[s] for s in stages], dtype=torch.long)

        return x.float(), y


def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])

    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B, T, 3000]
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)

    return xs, ys, lengths
