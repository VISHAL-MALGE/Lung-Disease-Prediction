import pandas as pd

# Load features and labels
features = pd.read_csv("dataset/features.csv")
labels = pd.read_csv("dataset/labels.csv")

# Debug print (optional) - helps you verify what's inside
print("Features Columns:", features.columns)
print("Labels Columns:", labels.columns)

# Merge based on correct column names
# Left: features["filename"] | Right: labels["image"]
merged = pd.merge(features, labels, left_on="filename", right_on="image", how="inner")

# Drop duplicate 'image' column (optional, as 'filename' is already there)
merged = merged.drop(columns=["image"])

# Save the final merged dataset
merged.to_csv("features_with_labels.csv", index=False)

print("✅ Merged features with labels successfully and saved to 'features_with_labels.csv'")
