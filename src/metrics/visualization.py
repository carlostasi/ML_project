import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.utils import print_log, get_model_folder_from_name, get_model_folder_from_model
import matplotlib
matplotlib.use('Agg')


def plot_decision_boundaries_2d(
    X_train,
    y_train,
    tuned_model,
    model_name,
    model_type,
    output_dir="results_notebook",
    big_data=False,
    random_state=42,
):
    """
    Draw a 2D decision boundary for the family of the tuned estimator.

    The estimator is duplicated from the tuned one with sklearn's clone, which
    copies every hyperparameter and returns an unfitted object. Rebuilding it
    from individually passed settings, as this function used to do, meant the
    plot could silently disagree with the model it claimed to show: it hardcoded
    gamma="scale" while the selected RBF SVM uses gamma='auto'. Cloning removes
    that whole class of drift, for every hyperparameter, permanently.

    What cloning cannot fix is that the plotted model is refitted on two
    principal components while the evaluated one sees all 38 features. This is
    a picture of how the model family carves up a projection of the space, not
    of the boundary any reported score was measured on, and the title says so.
    """
    print_log(f"Loading decision boundary plot for: {model_name}")
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    if tuned_model is None:
        raise ValueError("A fitted, tuned estimator is required to plot its boundary.")

    model = clone(tuned_model)
    model.fit(X_train_pca, y_train)

    x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
    y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5

    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#A0C4FF", "#CAFFBF", "#FFADAD"])
    cmap_bold = ["#001CF0", "#00B000", "#FF0000"]

    plt.figure(figsize=(9, 7))

    # Disegna le zone colorate di background
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading="auto")

    # Sovrappone i punti reali del dataset (ne stampiamo un subset per non affollare il grafico)
    rng = np.random.RandomState(random_state)
    subset_size = min(2000, len(X_train_pca))
    indices = rng.choice(len(X_train_pca), subset_size, replace=False)

    target_names = ["Low", "Medium", "High"]
    for class_idx, color in enumerate(cmap_bold):
        class_mask = (
            (y_train.iloc[indices] == class_idx)
            if hasattr(y_train, "iloc")
            else (y_train[indices] == class_idx)
        )
        plt.scatter(
            X_train_pca[indices][class_mask, 0],
            X_train_pca[indices][class_mask, 1],
            c=color,
            label=target_names[class_idx],
            edgecolor="k",
            s=25,
            alpha=0.7,
        )

    plt.title(
        f"Decision boundary - {model_name}\n"
        "refitted on 2 principal components, not the evaluated 38-feature model",
        fontsize=11,
    )
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal component 2 (PC2)")
    plt.legend(loc="upper right", title="Real classes")
    plt.tight_layout()

    folder = get_model_folder_from_name(model_name)
    final_output_dir = os.path.join(output_dir, folder) if folder else output_dir
    os.makedirs(final_output_dir, exist_ok=True)
    
    suffix = "_UsedBigData" if big_data else ""
    filepath = os.path.join(final_output_dir, f"decision_boundary_{model_type}{suffix}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved with success in: {filepath}")


def plot_correlation_matrix(df, output_dir="results_notebook", big_data=False):
    print(" Loading correlation matrix...")

    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 8},
    )

    plt.title("Correlation Matrix - Feature Space", fontsize=14, pad=20)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = "correlation_matrix_UsedBigData.png" if big_data else "correlation_matrix.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Correlation Matrix saved with success in: {filepath}")


def plot_feature_importance(
    model,
    feature_names,
    output_dir="results_notebook",
    filename="feature_importance.png",
    big_data=False
):
    if not hasattr(model, "feature_importances_"):
        print(
            f"[WARNING] Model {type(model).__name__} doesn't support the feature importance. Plot skipped."
        )
        return

    print(f"Generation Feature Importance for {type(model).__name__}...")
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(importances)), importances[indices], align="center", color="teal"
    )
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title(
        f"Feature Importance Matrix - {type(model).__name__}", fontsize=12, pad=15
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()

    folder = get_model_folder_from_model(model)
    final_output_dir = os.path.join(output_dir, folder) if folder else output_dir
    os.makedirs(final_output_dir, exist_ok=True)

    if big_data:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_UsedBigData{ext}"
        
    filepath = os.path.join(final_output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Feature Importance saved: {filepath}")


def plot_multiclass_roc(
    model,
    X_test,
    y_test,
    classes=["Low", "Medium", "High"],
    output_dir="results_notebook",
    filename="roc_curve.png",
    big_data=False
):
    print(f"Generation ROC curve for {type(model).__name__}...")

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(
            f"[WARNING] Model {type(model).__name__} doesn't support the calculation of the probabilities. ROC skipped."
        )
        return

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ["royalblue", "forestgreen", "crimson"]

    for i, color in enumerate(colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve: {classes[i]} (AUC = {roc_auc[i]:.4f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title(
        f"Receiver Operating Characteristic (OvR) - {type(model).__name__}",
        fontsize=12,
        pad=15,
    )
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    folder = get_model_folder_from_model(model)
    final_output_dir = os.path.join(output_dir, folder) if folder else output_dir
    os.makedirs(final_output_dir, exist_ok=True)

    if big_data:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_UsedBigData{ext}"

    filepath = os.path.join(final_output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"ROC Cruve saved: {filepath}")
