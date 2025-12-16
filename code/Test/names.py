import pandas as pd

# Load the two CSV files containing filenames in a column named 'Filename'
csv1 = pd.read_csv(r'D:\Sleep-Staging\filenames1.csv')
csv2 = pd.read_csv(r'D:\Sleep-Staging\csv-docs\cfs_visit5_selected.csv')

# Convert the filename columns to sets for set operations
set1 = set(csv1['Filename'])
set2 = set(csv2['path'])

# Find filenames in csv1 not in csv2
only_in_csv1 = set1 - set2

# Find filenames in csv2 not in csv1
only_in_csv2 = set2 - set1

# Convert to list or DataFrame if needed
only_in_csv1_list = list(only_in_csv1)
only_in_csv2_list = list(only_in_csv2)

print("Files in CSV1 not in CSV2:")
print(only_in_csv1_list)

print("Files in CSV2 not in CSV1:")
print(only_in_csv2_list)
