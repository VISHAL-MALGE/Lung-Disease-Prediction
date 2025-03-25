import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load features
features_df = pd.read_csv('dataset/features.csv')
features_df.columns = features_df.columns.str.strip()

# Load labels
labels_df = pd.read_csv('dataset/labels.csv')
labels_df.columns = labels_df.columns.str.strip()

# Rename 'image' column to 'filename'
labels_df.rename(columns={'image': 'filename'}, inplace=True)

# Debug - Check column names
print("Labels Columns after rename:", labels_df.columns)

# Merge on filename
data = pd.merge(features_df, labels_df, on='filename')

# Separate features and labels
X = data.drop(columns=['filename', 'label'])
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X, y_encoded)

# Save model and label encoder
joblib.dump(classifier, 'models/classifier.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("✅ Classifier and LabelEncoder saved successfully.")
