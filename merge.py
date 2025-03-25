import pandas as pd

# Load features and labels
features_df = pd.read_csv("features.csv")
labels_df = pd.read_csv("labels.csv")

# Ensure column names match for merging
features_df.rename(columns={"filename": "image"}, inplace=True)

# Merge on image name
merged_df = pd.merge(features_df, labels_df, on="image", how="left")

# Check for any missing labels
missing_labels = merged_df['label'].isnull().sum()
if missing_labels > 0:
    print(f"Warning: {missing_labels} features do not have corresponding labels!")

# Save the final features file with label column
merged_df.to_csv("features_with_labels.csv", index=False)

print("✅ Merged successfully! New file saved as 'features_with_labels.csv'")
