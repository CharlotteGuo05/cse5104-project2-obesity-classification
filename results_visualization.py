import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


results = pd.DataFrame(
    [
        ["Binary", "Full", "Random Forest", 0.985849],
        ["Binary", "Full", "SVM", 0.995283],
        ["Binary", "Full", "Decision Tree", 0.990566],
        ["Binary", "Full", "ANN", 0.985849],
        ["Multiclass", "Full", "Random Forest", 0.962264],
        ["Multiclass", "Full", "SVM", 0.938679],
        ["Multiclass", "Full", "Decision Tree", 0.919811],
        ["Multiclass", "Full", "ANN", 0.919811],
        ["Binary", "Reduced", "Random Forest", 0.985849],
        ["Binary", "Reduced", "SVM", 0.995283],
        ["Binary", "Reduced", "Decision Tree", 0.995283],
        ["Binary", "Reduced", "ANN", 0.985849],
        ["Multiclass", "Reduced", "Random Forest", 0.971698],
        ["Multiclass", "Reduced", "SVM", 0.948113],
        ["Multiclass", "Reduced", "Decision Tree", 0.952830],
        ["Multiclass", "Reduced", "ANN", 0.952830],
    ],
    columns=["Problem", "Feature Set", "Model", "Validation Accuracy"],
)

results["Experiment"] = results["Problem"] + " - " + results["Feature Set"]

sns.set_theme(style="whitegrid")
palette = {
    "Random Forest": "#2a9d8f",
    "SVM": "#264653",
    "Decision Tree": "#e9c46a",
    "ANN": "#e76f51",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for ax, problem in zip(axes, ["Binary", "Multiclass"]):
    subset = results[results["Problem"] == problem]
    sns.barplot(
        data=subset,
        x="Feature Set",
        y="Validation Accuracy",
        hue="Model",
        palette=palette,
        ax=ax,
    )
    ax.set_title(f"{problem} Classification", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Validation Accuracy" if problem == "Binary" else "")
    ax.set_ylim(0.88, 1.01)
    ax.tick_params(axis="x", labelsize=11)
    ax.legend(title="Model", loc="lower right", frameon=True)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

fig.suptitle(
    "Final Validation Accuracy by Model and Feature Set",
    fontsize=16,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig("model_results_validation_accuracy.png", dpi=300, bbox_inches="tight")
print("Saved model_results_validation_accuracy.png")

