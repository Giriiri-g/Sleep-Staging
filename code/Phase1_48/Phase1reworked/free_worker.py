import pandas as pd

df = pd.read_csv("processed_sleepedf/index.csv")

for idx, row in df.iterrows():
    tensor_path = row["tensor_path"]
    df.at[idx, "tensor_path"] = tensor_path.replace('processed_sleepedf\\tensors\\', 'home/geethalekshmy/GirishS/tensors/tensors')

df.to_csv("processed_sleepedf/index.csv", index=False)