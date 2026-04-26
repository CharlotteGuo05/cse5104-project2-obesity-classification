import pandas as pd
from sklearn.model_selection import train_test_split


# Split Binary Classification Dataset into Train & Validation
# Load binary classification dataset
print("Q3.1: SPLITTING BINARY CLASSIFICATION DATASET")
df = pd.read_csv("binary_classification_data.csv")

print(f"\nTotal dataset: {df.shape[0]} samples")
print(f"Class distribution:")
print(df['Obese_Binary'].value_counts())

# Split into features and target
X = df.drop(columns=['Obese_Binary'])
y = df['Obese_Binary']

# Split into training (90%) and validation (10%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,      # 10% for final validation
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"  - Not Obese (0): {(y_train == 0).sum()}")
print(f"  - Obese (1): {(y_train == 1).sum()}")
print(f"  - Class ratio: {(y_train == 1).mean():.1%} Obese")

print(f"\nValidation set: {X_val.shape[0]} samples")
print(f"  - Not Obese (0): {(y_val == 0).sum()}")
print(f"  - Obese (1): {(y_val == 1).sum()}")
print(f"  - Class ratio: {(y_val == 1).mean():.1%} Obese")

# Create train and validation DataFrames
train_df = X_train.copy()
train_df['Obese_Binary'] = y_train

val_df = X_val.copy()
val_df['Obese_Binary'] = y_val

# Save to CSV files
train_df.to_csv("binary_train.csv", index=False)
val_df.to_csv("binary_val.csv", index=False)

print("\n" + "-"*60)
print("Files saved:")
print("  binary_train.csv")
print("  binary_val.csv")
