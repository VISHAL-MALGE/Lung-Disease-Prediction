import os
import pandas as pd

# Paths
labels_csv_path = 'dataset/labels.csv'
images_folder_path = 'dataset/segmented_images'

# Load labels.csv
try:
    df = pd.read_csv(labels_csv_path)
except FileNotFoundError:
    print(f"❌ File not found: {labels_csv_path}")
    exit()

# Check if 'image' column exists
if 'image' not in df.columns:
    print("❌ 'image' column not found in labels.csv. Please check the file format.")
    exit()

# Track missing files
missing_files = []

# Check each image
for img in df['image']:
    img_path = os.path.join(images_folder_path, img)
    if not os.path.isfile(img_path):
        missing_files.append(img)

# Print results
if missing_files:
    print(f"❌ Total missing images: {len(missing_files)}")
    for img in missing_files:
        print(f"   - {img}")
else:
    print("✅ All images in labels.csv exist in the segmented_images folder.")
