import pandas as pd

# -------------------------
# FILE PATHS
# -------------------------
SLEEP_CSV = r"C:\Sleep-Staging\sleep_stages_output.csv"
CLINICAL_CSV = r"C:\Sleep-Staging\csv-docs\mesa_selected.csv"
OUTPUT_CSV = r"C:\Sleep-Staging\mesa_final.csv"

# -------------------------
# LOAD CSVs
# -------------------------
df_sleep = pd.read_csv(SLEEP_CSV)
df_clinical = pd.read_csv(CLINICAL_CSV)

# -------------------------
# EXTRACT MESAID FROM PATH
# -------------------------
df_clinical["mesaid"] = (
    df_clinical["path"]
    .str.extract(r"mesa-sleep-(\d+)\.edf", expand=False)
)

# Drop invalid rows
df_clinical = df_clinical.dropna(subset=["mesaid"])

# Convert to int
df_clinical["mesaid"] = df_clinical["mesaid"].astype(int)

# Ensure same type in sleep CSV
df_sleep["mesaid"] = df_sleep["mesaid"].astype(int)

# -------------------------
# SELECT REQUIRED COLUMNS
# -------------------------
cols_to_add = [
    "mesaid",
    "insomnia",
    "restless leg",
    "apnea",
    "snoring",
    "ahi_a0h3",
    "ai_all5",
    "odi35",
    "timest1p5",
    "timest2p5",
    "times34p5",
    "timeremp5",
    "slp_eff5",
    "slp_lat5",
    "waso5",
    "plmaslp5",
    "slpprdp5",
    "remlaiip5"
]

df_clinical = df_clinical[cols_to_add]

# -------------------------
# MERGE
# -------------------------
df_merged = pd.merge(
    df_sleep,
    df_clinical,
    on="mesaid",
    how="left"
)

# -------------------------
# SAVE
# -------------------------
df_merged.to_csv(OUTPUT_CSV, index=False)

print("Merge completed successfully.")
print("Total rows:", len(df_merged))
print("Unmatched mesaids:",
      df_merged[df_merged["insomnia"].isna()]["mesaid"].unique())
