# import pandas as pd
# filename = r'csv-docs\cfs-visit5-dataset-0.7.0 (1).csv'

# data = pd.read_csv(filename)
# count_above_60 = (data['age'] > 60).sum()

# df = pd.read_csv(r'csv-docs\wsc-dataset-0.7.0.csv')

# db = pd.read_csv(r'csv-docs\mros-visit1-dataset-0.6.0.csv')

# fd = pd.read_csv(r'csv-docs\shhs1-dataset-0.21.0.csv')
# # print([i for i in fd.columns if 'age' in i])
# database = pd.read_csv(r'csv-docs\apples-dataset-0.1.0.csv')


# print(f"CFS: {count_above_60}, total: {len(data)}")
# print(f"WSC: {(df['age'] > 60).sum()}, total: {len(df)}")
# print(f"MROS: {(db['vsage1'] > 60).sum()}, total: {len(db)}")
# print(f"SHHS: {(fd['age_s1'] > 60).sum()}, total: {len(fd)}")
# print(f"APPLES: {(database['age'] > 60).sum()}, total: {len(database)}")

import matplotlib.pyplot as plt
import numpy as np

# Data
studies = ["CFS", "WSC", "MROS", "SHHS", "APPLES", "MESA", "SLEEP-EDF"]
counts = [124, 1194, 2911, 3334, 336, 1805, 71]
totals = [735, 2570, 2911, 5804, 10612, 2237, 153]

# X positions
x = np.arange(len(studies))
width = 0.35  # bar width

# Create figure
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, counts, width, label='Age > 60')
bars2 = plt.bar(x + width/2, totals, width, label='Total')

# Add text labels above bars
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, height + max(totals)*0.01,
        f'{int(height)}', ha='center', va='bottom', fontsize=9
    )

# Labels and title
plt.xlabel("Study")
plt.ylabel("Number of Patients")
plt.title("Elder Patient Counts vs Total Records per Study")
plt.xticks(x, studies)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
