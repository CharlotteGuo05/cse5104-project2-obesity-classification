import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASS_NAMES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II",
]

CONFIGS = {
    ("binary", False): {
        "train_file": "binary_train.csv",
        "val_file": "binary_val.csv",
        "target_col": "Obese_Binary",
        "rf_n_estimators": 300,
        "dt_max_depth": 7,
        "ann_alpha": 1.0,
    },
    ("binary", True): {
        "train_file": "binary_train_reduced.csv",
        "val_file": "binary_val_reduced.csv",
        "target_col": "Obese_Binary",
        "rf_n_estimators": 150,
        "dt_max_depth": 9,
        "ann_alpha": 0.0001,
    },
    ("multiclass", False): {
        "train_file": "multiclass_train.csv",
        "val_file": "multiclass_val.csv",
        "target_col": "NObeyesdad",
        "rf_n_estimators": 300,
        "dt_max_depth": 11,
        "ann_alpha": 1.0,
    },
    ("multiclass", True): {
        "train_file": "multiclass_train_reduced.csv",
        "val_file": "multiclass_val_reduced.csv",
        "target_col": "NObeyesdad",
        "rf_n_estimators": 300,
        "dt_max_depth": 11,
        "ann_alpha": 0.001,
    },
}

# two inputs: binary vs multiclass, and reduced vs full features
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["binary", "multiclass"])
    parser.add_argument("--reduced", action="store_true")
    return parser.parse_args()

# load data based on the input arguments
def load_data(config):
    train = pd.read_csv(config["train_file"])
    vals = pd.read_csv(config["val_file"])
    x_train = train.drop(columns=[config["target_col"]])
    y_train = train[config["target_col"]]
    x_val = vals.drop(columns=[config["target_col"]])
    y_val = vals[config["target_col"]]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    return x_train, y_train, x_val, y_val, x_train_scaled, x_val_scaled

# build models based on the specifications in the config
def build_models(config, x_train, x_train_scaled, x_val, x_val_scaled):
    return {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=config["rf_n_estimators"], random_state=42, n_jobs=-1),
            "x_train": x_train,
            "x_val": x_val,
        },
        "SVM": {
            "model": SVC(kernel="linear", random_state=42),
            "x_train": x_train_scaled,
            "x_val": x_val_scaled,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=config["dt_max_depth"], random_state=42),
            "x_train": x_train,
            "x_val": x_val,
        },
        "ANN": {
            "model": MLPClassifier(alpha=config["ann_alpha"], max_iter=1000, random_state=42),
            "x_train": x_train_scaled,
            "x_val": x_val_scaled,
        },
    }

# evaluate each model and collect results
def evaluate_models(config, models, y_train, y_val):
    results = []
    for name, model_config in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training: {name}")

        start_train = time.time()
        model_config["model"].fit(model_config["x_train"], y_train)
        train_time = time.time() - start_train

        y_train_pred = model_config["model"].predict(model_config["x_train"])
        train_acc = accuracy_score(y_train, y_train_pred)

        start_test = time.time()
        y_pred = model_config["model"].predict(model_config["x_val"])
        test_time = time.time() - start_test

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted")
        recall = recall_score(y_val, y_pred, average="weighted")
        f1 = f1_score(y_val, y_pred, average="weighted")

        print("\nTRAINING RESULTS:")
        print(f"   Training Time: {train_time:.4f} seconds")
        print(f"   Training Accuracy: {train_acc:.4f}")

        print("\nTESTING RESULTS:")
        print(f"   Testing Time: {test_time:.4f} seconds")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        results.append(
            {
                "Model": name,
                "Training Time (s)": train_time,
                "Testing Time (s)": test_time,
                "Training Accuracy": train_acc,
                "Validation Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )

    return pd.DataFrame(results)


def main():
    args = parse_args()
    config = CONFIGS[(args.task, args.reduced)]
    x_train, y_train, x_val, y_val, x_train_scaled, x_val_scaled = load_data(config)
    models = build_models(config, x_train, x_train_scaled, x_val, x_val_scaled) 
    results_df = evaluate_models(config, models, y_train, y_val) # test models and collect results

    # print out best model based on validation accuracy
    best_idx = results_df["Validation Accuracy"].idxmax()
    best_model = results_df.loc[best_idx]
    print("=" * 70)
    print(f"\n BEST MODEL: {best_model['Model']}")
    print(f" Validation Accuracy: {best_model['Validation Accuracy']:.4f}")
    print(f" Training Time: {best_model['Training Time (s)']:.4f} seconds")
    print(f" Testing Time: {best_model['Testing Time (s)']:.4f} seconds")


if __name__ == "__main__":
    main()
