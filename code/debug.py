import pandas as pd

df_subjects = pd.read_excel('sleep-edf-database-expanded-1.0.0/SC-subjects.xls')
print(f"Loaded {len(df_subjects)} subject records from Excel file")
print(f"Columns: {list(df_subjects.columns)}")
print(df_subjects[0:1]['LightsOff'])  # Print a few rows for inspection