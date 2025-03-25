import cv2
import numpy as np
import mahotas
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    features = []

    try:
        # Resize
        image = cv2.resize(image, (128, 128))

        # Haralick Features (13)
        haralick = mahotas.features.haralick(image).mean(axis=0)
        features.extend(haralick)

        # Hu Moments (7)
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)

        # GLCM (5)
        glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
        glcm_features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0]
        ]
        features.extend(glcm_features)

        # ✨ Add 8 dummy zeros to make it 33 features
        while len(features) < 33:
            features.append(0.0)

        return np.array(features)

    except Exception as e:
        print(f"[Error] Feature extraction failed: {e}")
        return np.zeros(33)
