import pandas as pd

# Load original CSV
df = pd.read_csv(r"C:\Sleep-Staging\csv-docs\cfs-visit5-dataset-0.7.0 (1).csv")

# Create path column
df["path"] = (
    r"D:\cfs\files\polysomnography\edfs\cfs-visit5-" 
    + df["nsrrid"].astype(str) 
    + ".edf"
)

# Choose the columns you want (example: nsrrid, ecgdate, mob1 + path)
selected_cols = ["path","hrtdiag","angdiag","bpdiag", "cvdx", "diachol", "htfdiag", "irrdiag", "strodiag", "ischemia", "cerebdisease", "adddiag", "anxdiag", "behdiag", "depdiag", "htnx", "activeasthma", "brodiag", "haydiag", "pneudiag", "sindiag"]

# Create new dataframe with only chosen columns
df_selected = df[selected_cols]

# Save to new CSV
df_selected.to_csv(r"C:\Sleep-Staging\csv-docs\cfs_visit5_selected.csv", index=False)
