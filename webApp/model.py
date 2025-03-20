import cv2
import numpy as np
import random

def predict_disease(image_path):
    # Dummy model logic for now
    diseases = ["COPD", "Pulmonary Fibrosis", "Healthy Lungs"]
    lung_sides = ["Left Lung", "Right Lung", "Both Lungs"]
    descriptions = {
        "COPD": "Chronic Obstructive Pulmonary Disease affects airflow and breathing.",
        "Pulmonary Fibrosis": "Scarring of lung tissues that causes difficulty in breathing.",
        "Healthy Lungs": "No disease detected, lungs are in normal condition."
    }

    # Simulate prediction (Replace this with real ML model later)
    disease = random.choice(diseases)
    lung_side = random.choice(lung_sides)

    return disease, lung_side, descriptions[disease]
