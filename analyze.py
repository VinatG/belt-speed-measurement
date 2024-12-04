"""
Python script to analyse the speed logs.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_stats(speeds):
    stats = {"Mean": np.mean(speeds), "Median": np.median(speeds), "Standard Deviation": np.std(speeds), "1st Quartile (25%)": np.percentile(speeds, 25), "3rd Quartile (75%)": np.percentile(speeds, 75),
        "Interquartile Range (IQR)": np.percentile(speeds, 75) - np.percentile(speeds, 25), "Minimum": np.min(speeds), "Maximum": np.max(speeds), "Count": len(speeds),}
    return stats

# Load the data
file_path = "newwwww_speeds.csv"  # Path to the csv file
data = pd.read_csv(file_path)

column_name = "Smoothed Speed (mm/s)"

# Convert to numeric and drop NaN values
data[column_name] = pd.to_numeric(data[column_name], errors="coerce")
data = data.dropna(subset=[column_name])

speeds = data[column_name].to_numpy()

# Calculating the statistics
original_stats = calculate_stats(speeds)

print("\nStatistical Analysis Results:")
for key, value in original_stats.items():
    print(f"{key}: {value:.2f}")

# Boxplot
plt.figure(figsize = (8, 4))
plt.boxplot(speeds, vert = False, patch_artist = True, boxprops = dict(facecolor = "lightblue", linewidth = 1.5), medianprops = dict(color = "red", linewidth = 1.5))
plt.title("Original Boxplot of Smoothed Speeds")
plt.xlabel("Speed (mm/s)")
plt.savefig("boxplot_speeds.png")
plt.close()

# Histogram
plt.figure(figsize = (8, 4))
plt.hist(speeds, bins = 30, color = "lightgreen", edgecolor = "black", alpha = 0.7)
plt.title("Original Histogram of Smoothed Speeds")
plt.xlabel("Speed (mm/s)")
plt.ylabel("Frequency")
plt.savefig("histogram_speeds.png")
plt.close()

