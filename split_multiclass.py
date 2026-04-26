import pandas as pd
from sklearn.model_selection import train_test_split

# Split Multi-class Classification Dataset into Train & Validation
# Load multi-class classification dataset
print("Q4.1: SPLITTING MULTI-CLASS CLASSIFICATION DATASET")
df = pd.read_csv("multiclass_classification_data.csv")

print(f"\nTotal dataset: {df.shape[0]} samples")
print(f"Class distribution (NObeyesdad):")
print(df['NObeyesdad'].value_counts().sort_index())

# Split into features and target
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Split into training (90%) and validation (10%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,      # 10% for final validation
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Class distribution:")
print(y_train.value_counts().sort_index())

print(f"\nValidation set: {X_val.shape[0]} samples")
print(f"Class distribution:")
print(y_val.value_counts().sort_index())

# Create train and validation DataFrames
train_df = X_train.copy()
train_df['NObeyesdad'] = y_train

val_df = X_val.copy()
val_df['NObeyesdad'] = y_val

# Save to CSV files
train_df.to_csv("multiclass_train.csv", index=False)
val_df.to_csv("multiclass_val.csv", index=False)

print("\n" + "-"*60)
print("Files saved:")
print("multiclass_train.csv")
print("multiclass_val.csv")
