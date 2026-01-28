import pandas as pd
import torch


df = pd.read_csv("processed_sleepedf/index.csv")
sequence = df['stage_sequence'][0].split(" ")
file_path = df['tensor_path'][0]

x = torch.load(file_path)
print(f"Stage sequence: {len(sequence)} epochs")
print(f"Tensor shape: {x.shape}")  # (num_epochs, num_channels, num_samples)