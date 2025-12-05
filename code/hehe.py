import os
import csv

# Path to the folder
folder_path = r'D:\cfs\polysomnography\edfs'

# Get list of all filenames
filenames = os.listdir(folder_path)

# Write filenames to CSV
with open('filenames1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Filename'])
    # Write filenames
    for filename in filenames:
        writer.writerow([r"D:\cfs\files\polysomnography\edfs" + "\\"+filename])
