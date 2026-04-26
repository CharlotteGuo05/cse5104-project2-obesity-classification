import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# ============================================================
# Part C: Feature Selection for Dimension Reduction
# ============================================================

print("="*70)
print("PART C: FEATURE SELECTION FOR DIMENSION REDUCTION")
print("="*70)

# ============================================================
# Binary Classification Problem
# ============================================================
print("\n" + "="*70)
print("BINARY CLASSIFICATION: FEATURE SELECTION")
print("="*70)

# Load binary classification train and validation data
train_df = pd.read_csv("binary_train.csv")
val_df = pd.read_csv("binary_val.csv")

X_train = train_df.drop(columns=['Obese_Binary'])
y_train = train_df['Obese_Binary']
X_val = val_df.drop(columns=['Obese_Binary'])
y_val = val_df['Obese_Binary']

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"Features to keep: {X_train.shape[1] // 2} (half)")

# Apply feature selection
m = X_train.shape[1] // 2  # Keep half of features
selector = SelectKBest(score_func=mutual_info_classif, k=m)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

# Get selected feature names
selected_features_mask = selector.get_support()
selected_features = X_train.columns[selected_features_mask].tolist()

print(f"\nSelected features ({len(selected_features)}):")
print(selected_features)

print(f"\nFeature scores:")
for feature, score in zip(X_train.columns, selector.scores_):
    print(f"  {feature}: {score:.4f}")

# Create DataFrames with selected features
train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
train_selected_df['Obese_Binary'] = y_train.values

val_selected_df = pd.DataFrame(X_val_selected, columns=selected_features)
val_selected_df['Obese_Binary'] = y_val.values

# Save reduced datasets
train_selected_df.to_csv("binary_train_reduced.csv", index=False)
val_selected_df.to_csv("binary_val_reduced.csv", index=False)

print(f"\n✓ Saved: binary_train_reduced.csv ({train_selected_df.shape})")
print(f"✓ Saved: binary_val_reduced.csv ({val_selected_df.shape})")

# ============================================================
# Multi-class Classification Problem
# ============================================================
print("\n" + "="*70)
print("MULTI-CLASS CLASSIFICATION: FEATURE SELECTION")
print("="*70)

# Load multi-class classification train and validation data
train_df_mc = pd.read_csv("multiclass_train.csv")
val_df_mc = pd.read_csv("multiclass_val.csv")

X_train_mc = train_df_mc.drop(columns=['NObeyesdad'])
y_train_mc = train_df_mc['NObeyesdad']
X_val_mc = val_df_mc.drop(columns=['NObeyesdad'])
y_val_mc = val_df_mc['NObeyesdad']

print(f"\nOriginal features: {X_train_mc.shape[1]}")
print(f"Features to keep: {X_train_mc.shape[1] // 2} (half)")

# Apply feature selection
m_mc = X_train_mc.shape[1] // 2  # Keep half of features
selector_mc = SelectKBest(score_func=mutual_info_classif, k=m_mc)
X_train_mc_selected = selector_mc.fit_transform(X_train_mc, y_train_mc)
X_val_mc_selected = selector_mc.transform(X_val_mc)

# Get selected feature names
selected_features_mask_mc = selector_mc.get_support()
selected_features_mc = X_train_mc.columns[selected_features_mask_mc].tolist()

print(f"\nSelected features ({len(selected_features_mc)}):")
print(selected_features_mc)

print(f"\nFeature scores:")
for feature, score in zip(X_train_mc.columns, selector_mc.scores_):
    print(f"  {feature}: {score:.4f}")

# Create DataFrames with selected features
train_selected_df_mc = pd.DataFrame(X_train_mc_selected, columns=selected_features_mc)
train_selected_df_mc['NObeyesdad'] = y_train_mc.values

val_selected_df_mc = pd.DataFrame(X_val_mc_selected, columns=selected_features_mc)
val_selected_df_mc['NObeyesdad'] = y_val_mc.values

# Save reduced datasets
train_selected_df_mc.to_csv("multiclass_train_reduced.csv", index=False)
val_selected_df_mc.to_csv("multiclass_val_reduced.csv", index=False)

print(f"\n✓ Saved: multiclass_train_reduced.csv ({train_selected_df_mc.shape})")
print(f"✓ Saved: multiclass_val_reduced.csv ({val_selected_df_mc.shape})")

print("\n" + "="*70)
print("FEATURE SELECTION COMPLETE!")
print("="*70)