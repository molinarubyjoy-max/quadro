
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------
# STUDENT TRAINING DATA
# Features:
# 1. Study Hours
# 2. Attendance
# 3. Previous Score
# ------------------------------------------

X = [
    [1, 60, 40],
    [1, 65, 45],
    [2, 70, 50],
    [2, 75, 55],
    [3, 75, 60],
    [3, 80, 62],
    [4, 80, 65],
    [4, 85, 68],
    [5, 85, 70],
    [5, 90, 72],
    [6, 90, 75],
    [6, 92, 78],
    [7, 95, 80],
    [7, 96, 85],
    [8, 98, 90],
    [8, 99, 95],
    [2, 60, 45],
    [3, 70, 55],
    [5, 80, 65],
    [6, 85, 70],
]

# ------------------------------------------
# PERFORMANCE CATEGORIES
# ------------------------------------------

y = [
    "Needs Improvement",
    "Needs Improvement",
    "Needs Improvement",
    "Average",
    "Average",
    "Average",
    "Average",
    "Good",
    "Good",
    "Good",
    "Good",
    "Good",
    "Excellent",
    "Excellent",
    "Excellent",
    "Excellent",
    "Needs Improvement",
    "Average",
    "Good",
    "Good",
]

# ------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# ------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------

model_path = Path(__file__).parent / "model.pkl"

joblib.dump(model, model_path)

print("--------------------------------")
print("MODEL TRAINING COMPLETED")
print("--------------------------------")
print("Model saved successfully!")
print(f"File location: {model_path}")
