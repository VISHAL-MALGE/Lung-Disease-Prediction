# lung_segmentation.py (Final reviewed segment_lung function)

import os
import cv2
import numpy as np
import SimpleITK as sitk

# Paths
mask_folder = "dataset/seg-lungs-LUNA16"  # Path to .mhd mask folder

def load_mhd_mask(mask_path):
    """Load .mhd mask and convert to NumPy array."""
    itk_image = sitk.ReadImage(mask_path)
    return sitk.GetArrayFromImage(itk_image)

def segment_lung(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unreadable.")

    # Simple segmentation placeholder (you may already have your actual logic)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Optional: Determine affected side (dummy logic for now)
    height, width = thresh.shape
    left_lung = thresh[:, :width//2]
    right_lung = thresh[:, width//2:]

    left_intensity = np.sum(left_lung)
    right_intensity = np.sum(right_lung)

    if left_intensity < right_intensity:
        affected_side = "Left Lung"
    else:
        affected_side = "Right Lung"

    # Return only 2 values (Fix Here!)
    return thresh, affected_side
