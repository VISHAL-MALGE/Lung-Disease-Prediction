import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

# Load the merged dataset
data = pd.read_csv("features_with_labels.csv")

# Separate features and labels
X = data.drop(columns=["filename", "label"])
y = data["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Save the scaler in models folder
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train classifiers
models = {
    "knn": KNeighborsClassifier(),
    "svm": SVC(probability=True),
    "rf": RandomForestClassifier(),
    "dt": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    # ✅ Save the model in models folder
    with open(f"models/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅ {name.upper()} model trained and saved as models/{name}_model.pkl")

print("\n🎯 All models trained and saved successfully!")
