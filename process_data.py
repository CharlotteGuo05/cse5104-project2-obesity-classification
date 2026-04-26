import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

print("="*60)
print("DATASET OVERVIEW")
print(f"Total samples: {len(df)}")
print(f"\nOriginal target distribution (NObeyesdad):")
print(df["NObeyesdad"].value_counts())


categorical_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]

numerical_cols = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]

target_col = "NObeyesdad"

# two classification problems:
# Binary Classification: Obese vs Not Obese
print("\n" + "="*60)
print("\n1. Binary Classification Problem")

# Create binary target
obese_classes = ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
df["Obese_Binary"] = df[target_col].apply(lambda x: "Obese" if x in obese_classes else "Not Obese")

print("Class grouping:")
print(f"Obese: {obese_classes}")
print(f"Not Obese: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II")

print(f"\nClass distribution:")
print(df["Obese_Binary"].value_counts())

# features for binary classification
X_binary = df.drop(columns=[target_col, "Obese_Binary"])
y_binary = df["Obese_Binary"]
# Encode categorical features using one-hot encoding
X_binary_encoded = pd.get_dummies(X_binary, columns=categorical_cols, drop_first=True)
# Encode target: Obese=1, Not Obese=0
y_binary_encoded = (y_binary == "Obese").astype(int)

# Save binary classification dataset
binary_df = X_binary_encoded.copy()
binary_df["Obese_Binary"] = y_binary_encoded
binary_df.to_csv("binary_classification_data.csv", index=False)
print("\n Saved binary_classification_data.csv")


# Multi-class Classification: 7 Obesity Levels
print("\n2. Multi-class Classification Problem")
print("-" * 40)

# Use original 7-class target
y_multiclass = df[target_col]
print("Class distribution:")
print(y_multiclass.value_counts())

# features for multi-class classification
X_multiclass = df.drop(columns=[target_col, "Obese_Binary"])
y_multiclass = df[target_col]
# Encode categorical features using one-hot encoding
X_multiclass_encoded = pd.get_dummies(X_multiclass, columns=categorical_cols, drop_first=True)
# Encode target using LabelEncoder
le = LabelEncoder()
y_multiclass_encoded = le.fit_transform(y_multiclass)

print(f"\nClass mapping (label -> encoded value):")
for cls, val in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls}: {val}")

# Save multi-class classification dataset
multiclass_df = X_multiclass_encoded.copy()
multiclass_df["NObeyesdad"] = y_multiclass_encoded
multiclass_df.to_csv("multiclass_classification_data.csv", index=False)
print("\n Saved multiclass_classification_data.csv")


