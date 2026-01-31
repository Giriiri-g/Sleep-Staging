import pandas as pd
import numpy as np

# ============================
# USER INPUT
# ============================

CSV_PATH = r"csv-docs\mesa-sleep-dataset-0.8.0.csv"

# FEATURE_COLUMNS = [
#     'timest15',
#     'timest25',
#     'timest345',
#     'timerem5',
#     'waso5',
#     'timest1p5',
#     'timest2p5',
#     'times34p5',
#     'timeremp5',
#     'time_bed5',
#     'slpprdp5',
#     'slp_lat5',
#     'slp_eff5',
#     'remlaiip5'
# ]
FEATURE_COLUMNS = [
    # Identifiers & Demographics
    # "mesaid",
    "gender1",
    "sleepage5c",
    "bmi5c",

    # Sleep continuity & architecture
    "slp_lat5",
    "waso5",
    "slp_eff5",
    "time_bed5",
    "slpprdp5",
    "timest15",
    "timest25",
    "times34p5",
    "timerem5",
    "remlaiip5",

    # Respiratory / apnea physiology (PSG-derived)
    "oahi35",
    "oahi45",
    "ahiov505",
    "apnea35",
    "respevpr5"
]



DISORDER_COLUMNS = [
    'insmnia5',
    'rstlesslgs5',
    'slpapnea5'
]

ID_COLUMN = 'mesaid'

# ============================
# LOAD DATA
# ============================

df = pd.read_csv(CSV_PATH)
print(f"Loaded CSV with {len(df)} rows")

df_clean = df.copy()

# ============================
# FEATURE COLUMNS: NUMERIC + MEDIAN IMPUTATION
# ============================

for col in FEATURE_COLUMNS:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

df_clean[FEATURE_COLUMNS] = df_clean[FEATURE_COLUMNS].replace(
    [np.inf, -np.inf], np.nan
)

print("\n===== MISSING VALUES (FEATURES) BEFORE IMPUTATION =====")
print(df_clean[FEATURE_COLUMNS].isna().sum())

# Median imputation
for col in FEATURE_COLUMNS:
    median_value = df_clean[col].median()
    df_clean[col].fillna(median_value, inplace=True)

print("\n===== MISSING VALUES (FEATURES) AFTER IMPUTATION =====")
print(df_clean[FEATURE_COLUMNS].isna().sum())

# ============================
# DISORDER COLUMNS: FORCE BINARY + CLEAN
# ============================

for col in DISORDER_COLUMNS:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean[col] = df_clean[col].fillna(0)
    df_clean[col] = df_clean[col].astype(int)

print("\n===== DISORDER LABEL DISTRIBUTION =====")
for col in DISORDER_COLUMNS:
    print(f"\n{col}")
    print(df_clean[col].value_counts())

# ============================
# EXPORT FINAL DATASET
# ============================

EXPORT_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + DISORDER_COLUMNS

df_export = df_clean[EXPORT_COLUMNS].copy()

OUTPUT_PATH = "mesa_sleep_features_imputed_with_labels.csv"
df_export.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal dataset saved as: {OUTPUT_PATH}")
print(f"Exported columns:\n{EXPORT_COLUMNS}")
print(f"Final row count: {len(df_export)}")
