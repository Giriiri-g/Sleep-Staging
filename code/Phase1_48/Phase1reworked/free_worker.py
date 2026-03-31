import pandas as pd

# ==========================
# FILE PATHS
# ==========================
input_csv = r"C:\PS\Sleep-Staging\code\Phase1_48\Phase1reworked\output.csv"
output_csv = r"C:\PS\Sleep-Staging\code\Phase1_48\Phase1reworked\output.csv"

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(input_csv)

# ==========================
# REPLACE NULL / NaN VALUES
# ==========================
df = df.fillna(0)

# ==========================
# OPTIONAL: Replace inf values if present
# ==========================
df = df.replace([float('inf'), float('-inf')], 0)

# ==========================
# SAVE CLEANED CSV
# ==========================
df.to_csv(output_csv, index=False)

print("Cleaning complete")
print("Final shape:", df.shape)
print("Saved to:", output_csv)