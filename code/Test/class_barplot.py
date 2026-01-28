import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "mesa_final.csv"

DISORDER_COLUMNS = {
    "Insomnia": "insomnia",
    "Restless Leg Syndrome": "restless leg",
    "Sleep Apnea": "apnea"
}

FONT_SCALE = 1.5  # 50% increase in font size

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Count positive (1) and negative (0)
# -----------------------------
positive_counts = []
negative_counts = []

for disorder, col in DISORDER_COLUMNS.items():
    positive_counts.append((df[col] == 1).sum())
    negative_counts.append((df[col] == 0).sum())

# -----------------------------
# Plot
# -----------------------------
x = np.arange(len(DISORDER_COLUMNS))
width = 0.35

plt.figure(figsize=(12, 7))

bars_neg = plt.bar(
    x - width / 2,
    negative_counts,
    width,
    label="Negative (0)"
)

bars_pos = plt.bar(
    x + width / 2,
    positive_counts,
    width,
    label="Positive (1)"
)

# -----------------------------
# Labels & Styling
# -----------------------------
plt.title(
    "Positive vs Negative Class Counts for Sleep Disorders (MESA Dataset)",
    fontsize=16 * FONT_SCALE
)
plt.xlabel("Disorder", fontsize=12 * FONT_SCALE)
plt.ylabel("Number of Samples", fontsize=12 * FONT_SCALE)

plt.xticks(
    x,
    DISORDER_COLUMNS.keys(),
    fontsize=11 * FONT_SCALE
)
plt.yticks(fontsize=10 * FONT_SCALE)

plt.legend(fontsize=10 * FONT_SCALE)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# -----------------------------
# Value Annotations
# -----------------------------
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha="center",
            va="bottom",
            fontsize=9 * FONT_SCALE
        )

annotate_bars(bars_neg)
annotate_bars(bars_pos)

plt.tight_layout()
plt.show()
