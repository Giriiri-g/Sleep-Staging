import matplotlib.pyplot as plt

# Sleep stage labels and durations in seconds
stages = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'REM', 'Unscored']
seconds = [8578110, 645660, 2073960, 263790, 127380, 775050, 755250]

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(stages, seconds, color='skyblue')

# Add labels and title
plt.xlabel('Sleep Stage')
plt.ylabel('Duration (seconds)')
plt.title('Sleep Stage Duration Distribution')

# Display the exact seconds on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
