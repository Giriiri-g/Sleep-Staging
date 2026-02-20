import pandas as pd
import torch

df = pd.read_csv("C:/PS/Sleep-Staging/processed_sleepedf/index.csv")

epochs = []
for i in range(len(df)):
     epochs.append(torch.load(df.loc[i]['spectral']).shape[0])

mean = sum(epochs) / len(epochs)
print(mean)