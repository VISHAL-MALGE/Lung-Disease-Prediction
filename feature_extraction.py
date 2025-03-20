import os
import cv2
import numpy as np
import pandas as pd
import mahotas
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_hu

# Path to segmented images
input_folder = "Dataset/segmented_images"

# Check if directory exists
if not os.path.exists(input_folder):
    print(f"❌ Error: Directory '{input_folder}' not found!")
    exit()

# List all image files
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

if len(image_files) == 0:
    print("❌ No images found in the dataset folder. Check your preprocessing step!")
    exit()

print(f"✅ Found {len(image_files)} images in '{input_folder}'.")

features_list = []

# Gabor filter setup
def apply_gabor_filter(image):
    gabor_features = []
    ksize = 5  # Kernel size
    sigma = 1.0
    lambd = 10.0
    gamma = 0.5
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:  # Different orientations
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_features.append(filtered.mean())  # Mean response
        gabor_features.append(filtered.var())  # Variance response
    return gabor_features

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Error: Couldn't load {img_name}")
        continue

    # Extract Haralick features
    haralick_features = mahotas.features.haralick(img).mean(axis=0)

    # Extract GLCM features
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Extract Zernike moments
    hu_moments = moments_hu(img)

    # Extract Gabor features
    gabor_features = apply_gabor_filter(img)

    # Combine all features
    features = np.hstack([haralick_features, contrast, dissimilarity, homogeneity, energy, correlation, hu_moments, gabor_features])
    
    # Store features
    features_list.append([img_name] + list(features))

print(f"✅ Extracted features from {len(features_list)} images.")

# Convert to DataFrame
columns = ["filename"] + [f"feat_{i}" for i in range(len(features_list[0]) - 1)]
df = pd.DataFrame(features_list, columns=columns)

# Save as CSV
df.to_csv("Dataset/features.csv", index=False)
print("✅ Feature extraction complete! Features saved in 'Dataset/features.csv'")
