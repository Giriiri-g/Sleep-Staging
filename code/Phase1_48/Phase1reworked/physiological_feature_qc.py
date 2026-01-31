import pandas as pd
import numpy as np

# ============================
# USER INPUT
# ============================

CSV_PATH = r"csv-docs\mesa-sleep-dataset-0.8.0.csv"   # <-- put your CSV path here

COLUMNS_TO_CHECK = [
    # 'nsrr_tst_f1',
    # 'nsrr_tib_f1',
    # 'nsrr_ttleffsp_f1',
    # 'nsrr_ttllatsp_f1',
    # 'nsrr_waso_f1',
    # 'nsrr_pctdursp_s1',
    # 'nsrr_pctdursp_s2',
    # 'nsrr_pctdursp_s3',
    # 'nsrr_pctdursp_sr',
    # 'nsrr_ttlprdsp_s1sr',
    'timest15',
    'timest25',
    'timest345',
    'timerem5',
    'waso5',
    'timest1p5',
    'timest2p5',
    'times34p5',
    'timeremp5',
    'time_bed5',
    'slpprdp5',
    'slp_lat5',
    'slp_eff5',
    'remlaiip5'
]

# ============================
# LOAD DATA
# ============================

df = pd.read_csv(CSV_PATH)

print(f"Loaded CSV with {len(df)} rows")

# ============================
# FORCE NUMERIC CONVERSION
# Non-numeric values -> NaN
# ============================

df_numeric = df.copy()

for col in COLUMNS_TO_CHECK:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

# Replace inf / -inf with NaN
df_numeric[COLUMNS_TO_CHECK] = df_numeric[COLUMNS_TO_CHECK].replace(
    [np.inf, -np.inf], np.nan
)

# ============================
# PER-COLUMN ANALYSIS
# ============================

bad_per_column = df_numeric[COLUMNS_TO_CHECK].isna().sum()
good_per_column = df_numeric[COLUMNS_TO_CHECK].notna().sum()

column_report = pd.DataFrame({
    "total_rows": len(df_numeric),
    "good_entries": good_per_column,
    "bad_entries": bad_per_column,
    "bad_percentage": (bad_per_column / len(df_numeric)) * 100
})

print("\n===== COLUMN-WISE DATA QUALITY =====")
print(column_report)

# ============================
# ROW-WISE ANALYSIS
# ============================

# Row is GOOD only if ALL columns are valid
good_rows_mask = df_numeric[COLUMNS_TO_CHECK].notna().all(axis=1)

num_good_rows = good_rows_mask.sum()
num_bad_rows = len(df_numeric) - num_good_rows

print("\n===== ROW-WISE DATA QUALITY =====")
print(f"Total rows            : {len(df_numeric)}")
print(f"Fully valid rows      : {num_good_rows}")
print(f"Rows with any bad data: {num_bad_rows}")
print(f"Usable data (%)       : {(num_good_rows / len(df_numeric)) * 100:.2f}%")

# ============================
# OPTIONAL: SAVE CLEAN DATA
# ============================

# df_clean = df_numeric[good_rows_mask].reset_index(drop=True)

# df_clean.to_csv("cleaned_valid_rows.csv", index=False)
# print("\nClean dataset saved as: cleaned_valid_rows.csv")

# # ============================
# # OPTIONAL: SAVE REPORT
# # ============================

# column_report.to_csv("data_quality_report.csv")
# print("Data quality report saved as: data_quality_report.csv")
