import pandas as pd

# =============================
# Configuration
# =============================
EXCEL_PATH = "diseases_summary.xlsx"
OUTPUT_EXCEL = "dataset_variable_distributions.xlsx"

DATASETS = {
    "cfs": r"csv-docs\cfs-visit5-dataset-0.7.0 (1).csv",
    "apples": r"csv-docs\apples-dataset-0.1.0.csv",
    "mesa": r"csv-docs\mesa-sleep-dataset-0.8.0.csv",
    "shhs": r"csv-docs\shhs1-dataset-0.21.0.csv",
    "mros": r"csv-docs\mros-visit1-dataset-0.6.0.csv",
    "wsc": r"csv-docs\wsc-dataset-0.7.0.csv",
}

# =============================
# Load Excel metadata
# =============================
excel_df = pd.read_excel(
    EXCEL_PATH,
    sheet_name="Cleaned_list_with_names"
)

rows = []

# =============================
# Process datasets
# =============================
for dataset, csv_path in DATASETS.items():
    print(f"Processing {dataset.upper()}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load {csv_path}: {e}")
        continue

    id_col = f"{dataset} id".upper()
    name_col = f"{dataset.upper()} Name"
    desc_col = f"{dataset.upper()} Description"

    # Column name fixes
    if id_col == "APPLES ID":
        id_col = "Apples ID"
    if name_col == "CFS Name":
        name_col = "CFS NAME"

    if id_col not in excel_df.columns:
        continue

    for _, meta_row in excel_df.iterrows():
        var_id = meta_row[id_col]

        if pd.isna(var_id) or var_id not in df.columns:
            continue

        name = (
            meta_row[name_col]
            if name_col in excel_df.columns and not pd.isna(meta_row[name_col])
            else var_id
        )

        description = (
            meta_row[desc_col]
            if desc_col in excel_df.columns and not pd.isna(meta_row[desc_col])
            else ""
        )

        values = df[var_id].dropna()
        if values.empty:
            continue

        counts = values.value_counts()
        percentages = (counts / counts.sum() * 100).round(2)

        # Format strings
        percentage_str = ", ".join(
            [f"{k} ({v}%)" for k, v in percentages.items()]
        )

        count_str = ", ".join(
            [f"{k}: {v}" for k, v in counts.items()]
        )

        rows.append({
            "dataset": dataset.upper(),
            "id": var_id,
            "name": name,
            "description": description,
            "variables (percentages)": percentage_str,
            "true counts": count_str
        })

# =============================
# Export to Excel
# =============================
output_df = pd.DataFrame(rows)
output_df.to_excel(OUTPUT_EXCEL, index=False)

print(f"\nExport completed: {OUTPUT_EXCEL}")
