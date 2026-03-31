# import os
# import pandas as pd

# # ==========================
# # CONFIG
# # ==========================
# input_folder = r"C:\PS\Sleep-Staging\data\mesa\features"   # folder containing CSV files
# output_file = r"C:\PS\Sleep-Staging\code\Phase1_48\Phase1reworked\output.csv"  # final merged CSV

# # ==========================
# # LOAD AND MERGE CSV FILES
# # ==========================
# all_dataframes = []

# for file in os.listdir(input_folder):
#     if file.endswith(".csv"):
#         file_path = os.path.join(input_folder, file)
#         print(f"Reading: {file}")
        
#         df = pd.read_csv(file_path)
#         all_dataframes.append(df)

# # Stack all dataframes
# merged_df = pd.concat(all_dataframes, axis=0, ignore_index=True)

# # Save merged CSV
# merged_df.to_csv(output_file, index=False)

# print("=================================")
# print(f"Merged {len(all_dataframes)} files")
# print(f"Final shape: {merged_df.shape}")
# print(f"Saved to: {output_file}")
# print("=================================")

import pandas as pd

# ==========================
# FILE PATHS
# ==========================
output_csv = r"C:\PS\Sleep-Staging\code\Phase1_48\Phase1reworked\output.csv"
mesa_csv = r"C:\PS\Sleep-Staging\code\Phase1_48\Phase1reworked\mesa_sleep_features_imputed_with_labels.csv"
final_output = output_csv

# ==========================
# LOAD FILES
# ==========================
df_output = pd.read_csv(output_csv)
df_mesa = pd.read_csv(mesa_csv)

# ==========================
# SELECT REQUIRED COLUMNS
# ==========================
mesa_subset = df_mesa[[
    "mesaid",
    "insmnia5",
    "rstlesslgs5",
    "slpapnea5"
]]

# ==========================
# RENAME COLUMNS
# ==========================
mesa_subset = mesa_subset.rename(columns={
    "insmnia5": "Insomnia",
    "rstlesslgs5": "RLS",
    "slpapnea5": "apnea"
})

# ==========================
# MERGE DATASETS
# ==========================
merged = df_output.merge(
    mesa_subset,
    left_on="nsrrid",
    right_on="mesaid",
    how="left"
)

# Remove duplicate key column
merged = merged.drop(columns=["mesaid"])

# ==========================
# SAVE RESULT
# ==========================
merged.to_csv(final_output, index=False)

print("Merge complete")
print("Final shape:", merged.shape)
print("Saved to:", final_output)