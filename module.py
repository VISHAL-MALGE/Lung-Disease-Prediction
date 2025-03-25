import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Create models directory if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Step 1: Load features and labels
print("\n[INFO] Loading features.csv...")
data = pd.read_csv("Dataset/features.csv")

# Separate labels
y = data.iloc[:, -1].values  # Last column is target label

# Step 1.5: Convert continuous labels into categorical classes
print("[INFO] Converting continuous labels to categorical classes...")
bins = [0, 1.5, 2.5, 3.5, 4.5, np.inf]
labels = ['COPD', 'Pulmonary Fibrosis', 'Healthy Lungs', 'Lung Cancer', 'Pneumonia']
y = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

# Show label mapping preview
print("\n[INFO] Label Mapping Preview:")
preview_df = pd.DataFrame({'Original Continuous Value': data.iloc[:, -1], 'Mapped Class': y})
print(preview_df.head(10))  # Show first 10 rows

# Drop non-numeric columns (e.g., filenames) and target column from X
X = data.select_dtypes(include=[np.number]).drop(columns=[data.columns[-1]]).values

# Step 2: Standardize features
print("[INFO] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Feature selection using PCA
print("[INFO] Applying PCA for feature selection...")
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# Step 4: Train-Test Split with smaller subset for testing
print("[INFO] Splitting data into train and test sets...")
X_pca_small = X_pca[:10000]
y_small = y[:10000]
X_train, X_test, y_train, y_test = train_test_split(X_pca_small, y_small, test_size=0.2, random_state=42)

# Step 5: Train Models
print("\n[INFO] Training Models...")

# KNN
print("Training KNN...")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, "models/knn_model.pkl")

# SVM
print("Training SVM...")
svm = SVC(probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "models/svm_model.pkl")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "models/rf_model.pkl")

# Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
joblib.dump(dt, "models/dt_model.pkl")

# Step 6: Evaluation Function
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Evaluate All Models
evaluate_model(knn, "KNN")
evaluate_model(svm, "SVM")
evaluate_model(rf, "Random Forest")
evaluate_model(dt, "Decision Tree")

print("\n[INFO] Model training and evaluation complete.")
