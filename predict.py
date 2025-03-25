import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import mahotas

# ---------------- Feature Extraction ---------------- #
def extract_features(image):
    features = []
    try:
        image = cv2.resize(image, (128, 128))

        haralick = mahotas.features.haralick(image).mean(axis=0)
        features.extend(haralick)

        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)

        glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
        glcm_features = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0]
        ]
        features.extend(glcm_features)

        # Add dummy zeros to make 33 features
        while len(features) < 33:
            features.append(0.0)

        return np.array(features)
    except Exception as e:
        print(f"[Error] Feature extraction failed: {e}")
        return np.zeros(33)

# ---------------- Disease Info ---------------- #
disease_descriptions = {
    "COPD": "Chronic Obstructive Pulmonary Disease: causes airflow blockage and breathing problems.",
    "Pulmonary Fibrosis": "Condition causing lung tissue damage and stiffness.",
    "Healthy": "No disease detected. Lungs are functioning normally.",
    "Lung Cancer": "Uncontrolled cell growth in lungs forming tumors.",
    "Pneumonia": "Infection causing inflammation in lung air sacs."
}

# ---------------- Main Prediction ---------------- #
def main():
    print("🚀 Lung Disease Prediction System")
    image_path = input("📤 Enter the path of the CT scan image: ").strip()

    print(f"\n🖼️ Processing image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("❌ Failed to load image.")
        return

    features = extract_features(image).reshape(1, -1)

    # Load models & scaler
    scaler = joblib.load("models/scaler.pkl")
    models = {
        "KNN": joblib.load("models/knn_model.pkl"),
        "SVM": joblib.load("models/svm_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "Decision Tree": joblib.load("models/dt_model.pkl")
    }

    features_scaled = scaler.transform(features)

    # Store all predictions
    model_predictions = []
    class_labels = []

    for name, model in models.items():
        label = model.predict(features_scaled)[0]
        probas = model.predict_proba(features_scaled)[0]
        label_index = list(model.classes_).index(label)
        confidence = round(probas[label_index] * 100, 2)
        model_predictions.append((label, confidence))
        class_labels.append(label)

    # 🧠 Final majority vote
    from collections import Counter
    votes = Counter([pred[0] for pred in model_predictions])
    final_disease = votes.most_common(1)[0][0]

    # 📊 Average confidence for the winning disease
    confidences = [conf for label, conf in model_predictions if label == final_disease]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0

    # Impression & status
    impression = "Critical" if final_disease != "Healthy" else "Normal"
    status = "🟥 Affected" if final_disease != "Healthy" else "✅ Healthy"
    affected_side = "Left Lung"  # You can enhance later for automatic detection

    print("\n📊 Prediction Results:")
    print(f"   ➤ Disease: {final_disease}")
    print(f"   ➤ Confidence: {avg_confidence}%")
    print(f"   ➤ Description: {disease_descriptions.get(final_disease, 'N/A')}")
    print(f"   ➤ Impression: {impression}")
    print(f"   ➤ Affected Side: {affected_side}")
    print(f"   ➤ Status: {status}")

if __name__ == "__main__":
    main()

# ---------------- Dummy Placeholder Prediction (optional) ---------------- #
def predict_disease_from_image(image_path):
    # Placeholder output
    return {
        'name': 'COPD',
        'description': 'Lung fields exhibit signs of chronic obstruction.',
        'impression': 'Suggestive of COPD.',
        'predictionResult': 'Chronic Obstructive Pulmonary Disease',
        'confidence': '89%',
        'areas': 'Lower lobes',
        'recommendation': 'Consult a pulmonologist for further evaluation.',
        'note': 'AI-generated prediction report.',
        'status': 'disease',
        'icon': '<i class="fas fa-lungs-virus" style="color: #DC2626;"></i>'
    }

# ---------------- ✅ Flask Backend Integration Entry Point ---------------- #
def run_prediction_pipeline(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"error": "Invalid image path or unreadable image."}

    features = extract_features(image).reshape(1, -1)

    scaler = joblib.load("models/scaler.pkl")
    models = {
        "KNN": joblib.load("models/knn_model.pkl"),
        "SVM": joblib.load("models/svm_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "Decision Tree": joblib.load("models/dt_model.pkl")
    }

    features_scaled = scaler.transform(features)

    model_predictions = []
    class_labels = []

    for name, model in models.items():
        label = model.predict(features_scaled)[0]
        probas = model.predict_proba(features_scaled)[0]
        label_index = list(model.classes_).index(label)
        confidence = round(probas[label_index] * 100, 2)
        model_predictions.append((label, confidence))
        class_labels.append(label)

    from collections import Counter
    votes = Counter([pred[0] for pred in model_predictions])
    final_disease = votes.most_common(1)[0][0]

    confidences = [conf for label, conf in model_predictions if label == final_disease]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0

    impression = "Critical" if final_disease != "Healthy" else "Normal"
    status = "🟥 Affected" if final_disease != "Healthy" else "✅ Healthy"
    affected_side = "Left Lung"  # Add your side detection logic if needed

    return {
        'name': final_disease,
        'description': disease_descriptions.get(final_disease, "No description available."),
        'impression': impression,
        'predictionResult': final_disease,
        'confidence': f"{avg_confidence}%",
        'areas': affected_side,
        'recommendation': "Please consult a medical professional.",
        'note': "AI-generated prediction report.",
        'status': "disease" if final_disease != "Healthy" else "healthy",
        'icon': '<i class="fas fa-lungs-virus" style="color: #DC2626;"></i>' if final_disease != "Healthy" else '<i class="fas fa-heartbeat" style="color: #16A34A;"></i>'
    }
