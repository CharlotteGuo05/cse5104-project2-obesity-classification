import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")

# specify configurations for different tasks and feature sets
CONFIGS = {
    ("binary", False): {
        "train_file": "binary_train.csv",
        "target_col": "Obese_Binary",
        "output_file": "cv_results_visualization.png",
        "figure_title": "K-Fold Cross-Validation Results for Different Models",
        "load_message": "Loading binary training data...",
    },
    ("multiclass", False): {
        "train_file": "multiclass_train.csv",
        "target_col": "NObeyesdad",
        "output_file": "cv_results_visualization_multiclass.png",
        "figure_title": "K-Fold Cross-Validation Results for Different Models (Multi-class)",
        "load_message": "Loading multi-class training data...",
    },
    ("binary", True): {
        "train_file": "binary_train_reduced.csv",
        "target_col": "Obese_Binary",
        "output_file": "cv_results_visualization_reduced.png",
        "figure_title": "K-Fold Cross-Validation Results for Different Models (Reduced Features)",
        "load_message": "Loading REDUCED training data...",
    },
    ("multiclass", True): {
        "train_file": "multiclass_train_reduced.csv",
        "target_col": "NObeyesdad",
        "output_file": "cv_results_visualization_multiclass_reduced.png",
        "figure_title": "K-Fold Cross-Validation Results for Different Models (Multi-class, Reduced Features)",
        "load_message": "Loading REDUCED multi-class training data...",
    },
}

# specify model specifications for grid search and plotting
MODEL_SPECS = [
    {
        "name": "Random Forest",
        "param_name": "n_estimators",
        "display_name": "n_estimators",
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 150, 200, 300]},
        "use_scaled": False,
        "plot_kind": "line",
        "x_label": "Number of Trees (n_estimators)",
        "title": "Random Forest: Number of Trees",
    },
    {
        "name": "SVM",
        "param_name": "kernel",
        "display_name": "kernel",
        "model": SVC(random_state=42),
        "params": {"kernel": ["linear", "poly", "rbf"]},
        "use_scaled": True,
        "plot_kind": "bar",
        "x_label": "Kernel Type",
        "title": "SVM: Kernel Type",
    },
    {
        "name": "Decision Tree",
        "param_name": "max_depth",
        "display_name": "max_depth",
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [3, 5, 7, 9, 11]},
        "use_scaled": False,
        "plot_kind": "line",
        "x_label": "Maximum Depth",
        "title": "Decision Tree: Maximum Depth",
    },
    {
        "name": "ANN",
        "param_name": "alpha",
        "display_name": "alpha",
        "model": MLPClassifier(max_iter=1000, random_state=42),
        "params": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
        "use_scaled": True,
        "plot_kind": "line",
        "x_label": "Regularization Parameter (alpha) - log scale",
        "title": "ANN: Regularization Strength",
        "x_log": True,
    },
]

# two inputs: binary vs multiclass, and reduced vs full features
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["binary", "multiclass"])
    parser.add_argument("--reduced", action="store_true")
    return parser.parse_args()

# load data based on the input arguments, and print dataset info if specified in config
def load_data(config):
    print(config["load_message"])
    train = pd.read_csv(config["train_file"])
    x_train = train.drop(columns=[config["target_col"]])
    y_train = train[config["target_col"]]

    # scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    return x_train, y_train, x_train_scaled

# run grid search
def run_grid_search(model, params, x_train, y_train):
    grid = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(x_train, y_train)
    results = pd.DataFrame(grid.cv_results_)
    param_name = next(iter(params))
    param_values = params[param_name]
    means = results["mean_test_score"]
    stds = results["std_test_score"]
    best_idx = means.idxmax()
    return param_values, means, stds, best_idx

# function to plot line graph with error bars
def plot_line_axis(ax, x_values, means, stds, spec):
    ax.errorbar(x_values,means,yerr=stds,capsize=5,
                linewidth=2,markersize=8,color="#1f77b4",ecolor="#1f77b4")
    ax.set_xlabel(spec["x_label"], fontsize=11)
    ax.set_ylabel("Cross-Validation Accuracy", fontsize=11)
    ax.set_title(spec["title"], fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if spec.get("x_log"):
        ax.set_xscale("log")

# function to plot bar graph with error bars
def plot_bar_axis(ax, x_values, means, stds, spec):
    x_pos = np.arange(len(x_values))
    ax.bar(x_pos,means,yerr=stds,capsize=5,color="#1f77b4",ecolor="#1f77b4",
           edgecolor="black",linewidth=1,)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values, fontsize=11)
    ax.set_xlabel(spec["x_label"], fontsize=11)
    ax.set_ylabel("Cross-Validation Accuracy", fontsize=11)
    ax.set_title(spec["title"], fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.005, f"{mean:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# mark the best value on the plot
def annotate_best(ax, x_values, means, best_idx, spec):
    x_best = x_values[best_idx]
    y_best = means[best_idx]
    if spec["name"] == "Random Forest":
        label = f"Best: {x_best} trees\nAcc: {y_best:.4f}"
    elif spec["name"] == "Decision Tree":
        label = f"Best: depth={x_best}\nAcc: {y_best:.4f}"
    elif spec["name"] == "ANN":
        label = f"Best: alpha={x_best}\nAcc: {y_best:.4f}"
    else:
        return # svm doesn't need this
    ax.annotate(label,xy=(x_best, y_best),xytext=(10, 10),textcoords="offset points",fontsize=9)

# print out best hyperparameter in command line
def print_best_line(spec, x_values, means, best_idx):
    best_value = x_values[best_idx]
    best_score = means[best_idx]
    if spec["name"] == "Random Forest":
        print(f"Best n_estimators: {best_value}, Accuracy: {best_score:.4f}")
    elif spec["name"] == "SVM":
        print(f"Best kernel: {best_value}, Accuracy: {best_score:.4f}")
    elif spec["name"] == "Decision Tree":
        print(f"Best max_depth: {best_value}, Accuracy: {best_score:.4f}")
    elif spec["name"] == "ANN":
        print(f"Best alpha: {best_value}, Accuracy: {best_score:.4f}")


def main():
    args = parse_args()
    config = CONFIGS[(args.task, args.reduced)]

    x_train, y_train, x_train_scaled = load_data(config) # load data based on input arguments

    print("\nRunning Grid Search with 5-Fold Cross-Validation for each model...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(config["figure_title"], fontsize=16, fontweight="bold")

    summary_rows = []
    flat_axes = axes.flatten()
    # loop through 4 models and plot results
    for idx, spec in enumerate(MODEL_SPECS):
        print(f"\n Training {spec['name']} ...")
        model_x = x_train_scaled if spec["use_scaled"] else x_train
        param_values, means, stds, best_idx = run_grid_search(spec["model"], spec["params"], model_x, y_train)

        ax = flat_axes[idx]
        if spec["plot_kind"] == "bar":
            plot_bar_axis(ax, param_values, means, stds, spec)
        else:
            plot_line_axis(ax, param_values, means, stds, spec)
            annotate_best(ax, param_values, means, best_idx, spec)
        
        print_best_line(spec, param_values, means, best_idx)

    plt.tight_layout()
    plt.savefig(config["output_file"], dpi=300, bbox_inches="tight")
    print(f"\n plot saved as '{config['output_file']}'")


if __name__ == "__main__":
    main()
