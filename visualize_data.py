import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('dataset/master_yt_bb_detection.csv', header=None)

# Column 3 contains the class labels
class_column = 3
class_counts = df[class_column].value_counts()

# Create figure with appropriate size
plt.figure(figsize=(14, 8))

# Create a bar plot
bars = plt.bar(range(len(class_counts)), class_counts.values, color='steelblue', edgecolor='navy', alpha=0.7)

# Customize the plot
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Distribution of Classes in YouTube Bounding Box Detection Dataset Per Frame', fontsize=14, fontweight='bold')

# Set x-axis labels
plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, class_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000, 
             f'{value:,}', ha='center', va='bottom', fontsize=9)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('Class_distribution.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'Class_distribution.png'")

# Display the plot
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("CLASS DISTRIBUTION SUMMARY")
print("="*60)
print(f"\nTotal number of samples: {len(df):,}")
print(f"Number of unique classes: {len(class_counts)}")
print(f"\nClass-wise breakdown:")
print(f"{'Class':<20} {'Count':<15} {'Percentage':<15}")
print("-" * 50)
for class_name, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{class_name:<20} {count:<15,} {percentage:>6.2f}%")
print("="*60)