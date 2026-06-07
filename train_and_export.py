import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Set paths
DATA_PATH = "student.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Preserve a copy of raw data for extracting UI sample profiles later
df_raw = df.copy()

# 2. Preprocess & Encode Features
print("Preprocessing and encoding features...")

# Drop columns matching the Jupyter notebook
df = df.drop(columns=["NationalITy", "PlaceofBirth"])

# Map ordinal features
df['StageID'] = df['StageID'].map({
    'lowerlevel': 0,
    'MiddleSchool': 1,
    'HighSchool': 2
})

# Clean GradeID
df['GradeID'] = df['GradeID'].str.replace('G-', '').astype(int)

# Binary/Label Encodings (alphabetic mappings consistent with LabelEncoder)
df['gender_encoded'] = df['gender'].map({'F': 0, 'M': 1})
df = df.drop(columns=['gender'])

df['Semester_encoded'] = df['Semester'].map({'F': 0, 'S': 1})
df = df.drop(columns=['Semester'])

df['Relation_encoded'] = df['Relation'].map({'Father': 0, 'Mum': 1})
df = df.drop(columns=['Relation'])

df['ParentAnsweringSurvey_encoded'] = df['ParentAnsweringSurvey'].map({'No': 0, 'Yes': 1})
df = df.drop(columns=['ParentAnsweringSurvey'])

df['ParentschoolSatisfaction_encoded'] = df['ParentschoolSatisfaction'].map({'Bad': 0, 'Good': 1})
df = df.drop(columns=['ParentschoolSatisfaction'])

df['StudentAbsenceDays_encoded'] = df['StudentAbsenceDays'].map({'Above-7': 0, 'Under-7': 1})
df = df.drop(columns=['StudentAbsenceDays'])

# One-hot encodes SectionID and Topic
SectionID_hotenc = pd.get_dummies(df["SectionID"], prefix="SectionID", dtype=int)
df = pd.concat([df, SectionID_hotenc], axis=1).drop(columns=["SectionID"])

Topic_hotenc = pd.get_dummies(df["Topic"], prefix="Topic", dtype=int)
df = pd.concat([df, Topic_hotenc], axis=1).drop(columns=["Topic"])

# Define numeric columns
numeric_columns = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

# Separate features and target
X = df.drop(columns=["Class"])
y = df['Class'].map({'L': 0, 'M': 1, 'H': 2})

# Save the exact list of final columns for app.py
feature_columns = list(X.columns)
with open(os.path.join(MODELS_DIR, "columns.json"), "w") as f:
    json.dump(feature_columns, f, indent=4)

# 3. Train-Test Split (80/20 Stratified)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
print("Scaling numeric features...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
print("Scaler saved successfully.")

# 5. Train Models
print("Training models...")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Random Forest (Primary Model)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# Save Random Forest model
joblib.dump(rf, os.path.join(MODELS_DIR, "model.joblib"))
print("Random Forest model saved successfully.")

# 6. Evaluation & Metrics
print("Evaluating models...")

def get_metrics(model, X_tr, X_te, y_tr, y_te):
    pred = model.predict(X_te)
    cm = confusion_matrix(y_te, pred).tolist()
    return {
        "training_accuracy": float(model.score(X_tr, y_tr)),
        "testing_accuracy": float(model.score(X_te, y_te)),
        "precision": float(precision_score(y_te, pred, average="weighted")),
        "recall": float(recall_score(y_te, pred, average="weighted")),
        "f1_score": float(f1_score(y_te, pred, average="weighted")),
        "confusion_matrix": cm
    }

metrics = {
    "KNN": get_metrics(knn, X_train_scaled, X_test_scaled, y_train, y_test),
    "LogisticRegression": get_metrics(lr, X_train_scaled, X_test_scaled, y_train, y_test),
    "RandomForest": get_metrics(rf, X_train_scaled, X_test_scaled, y_train, y_test)
}

# Write metrics
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print("Performance metrics written to metrics.json.")

# 7. Extract Feature Importances (Random Forest)
importances = rf.feature_importances_
feat_importances = []
for col, imp in zip(X.columns, importances):
    feat_importances.append({
        "feature": col,
        "importance": float(imp)
    })
# Sort by importance descending
feat_importances = sorted(feat_importances, key=lambda x: x["importance"], reverse=True)

with open(os.path.join(MODELS_DIR, "feature_importances.json"), "w") as f:
    json.dump(feat_importances, f, indent=4)
print("Feature importances written.")

# 8. Export Sample Profiles for UI Loader
# Pick 4 clear representative students from the original dataset
# Index 0: Low performance student
# Index 10: Medium performance student
# Index 49: High performance student
# Index 25: Low performance student
sample_indices = [3, 11, 49, 134]  # Chosen from raw CSV records
sample_profiles = []

for idx in sample_indices:
    row = df_raw.iloc[idx]
    sample_profiles.append({
        "name": f"Sample Profile {len(sample_profiles)+1} ({row['Class']} Class)",
        "Class": str(row['Class']),
        "gender": str(row['gender']),
        "StageID": str(row['StageID']),
        "GradeID": str(row['GradeID']),
        "SectionID": str(row['SectionID']),
        "Topic": str(row['Topic']),
        "Semester": str(row['Semester']),
        "Relation": str(row['Relation']),
        "raisedhands": int(row['raisedhands']),
        "VisITedResources": int(row['VisITedResources']),
        "AnnouncementsView": int(row['AnnouncementsView']),
        "Discussion": int(row['Discussion']),
        "ParentAnsweringSurvey": str(row['ParentAnsweringSurvey']),
        "ParentschoolSatisfaction": str(row['ParentschoolSatisfaction']),
        "StudentAbsenceDays": str(row['StudentAbsenceDays'])
    })

with open(os.path.join(MODELS_DIR, "samples.json"), "w") as f:
    json.dump(sample_profiles, f, indent=4)
print("Representative samples written.")

print("All training and export tasks completed successfully!")
