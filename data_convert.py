import os
import SimpleITK as sitk
import numpy as np
import cv2

# Define input and output directories
input_folder = "Dataset\\subset1"  # Change this if needed
output_folder = "Dataset\\processed_images"
os.makedirs(output_folder, exist_ok=True)

# Convert all .mhd files
for file in os.listdir(input_folder):
    if file.endswith(".mhd"):
        file_path = os.path.join(input_folder, file)
        
        # Read the .mhd image
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)  # Shape: (slices, height, width)

        # Save each slice as a separate image
        for i in range(image_array.shape[0]):
            img_slice = image_array[i]
            img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX)
            img_slice = np.uint8(img_slice)
            
            output_path = os.path.join(output_folder, f"{file.replace('.mhd', '')}_slice_{i}.png")
            cv2.imwrite(output_path, img_slice)

print("✅ Conversion complete! Images saved in:", output_folder)
