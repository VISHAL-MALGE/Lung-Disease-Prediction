import os
import cv2
import numpy as np
import SimpleITK as sitk  # For reading .mhd files
from tqdm import tqdm

# Paths
input_folder = "Dataset/processed_images"  # PNG images
mask_folder = "Dataset/seg-lungs-LUNA16"  # Segmentation masks (.mhd format)
output_folder = "Dataset/segmented_images"  # Save segmented lungs

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

def load_mhd_mask(mask_path):
    """ Load .mhd mask and convert it to a NumPy array """
    itk_image = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(itk_image)  # Shape: (depth, height, width)
    return mask_array  # 3D mask (slices)

# Get available masks (excluding .zraw files)
available_masks = {f.split(".mhd")[0] for f in os.listdir(mask_folder) if f.endswith(".mhd")}

# Process each image
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".png"):  # Ensure it's an image file
        image_path = os.path.join(input_folder, filename)
        
        # Extract scan ID from filename
        scan_id = filename.split("_slice_")[0]  # Removes '_slice_X.png'
        slice_number = int(filename.split("_slice_")[-1].replace(".png", ""))

        # Find closest matching mask file
        if scan_id in available_masks:
            mask_file = os.path.join(mask_folder, f"{scan_id}.mhd")

            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Load corresponding .mhd mask
            mask_array = load_mhd_mask(mask_file)

            if slice_number < mask_array.shape[0]:  # Check if slice exists in the mask
                mask = mask_array[slice_number]  # Get corresponding slice
                mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask

                # Apply mask to segment lungs
                segmented = cv2.bitwise_and(image, image, mask=mask)

                # Save segmented image
                cv2.imwrite(os.path.join(output_folder, filename), segmented)
            else:
                print(f"❌ Mask slice {slice_number} not found in {mask_file}")
        else:
            print(f"❌ No matching mask file for: {filename}")

print("✅ Lung segmentation complete! Segmented images saved in:", output_folder)
