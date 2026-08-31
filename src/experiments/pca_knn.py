"""
Does dimensionality reduction rescue K-NN?
"""

import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from src.models import load_model, model_exists
from src.utils import print_log

matplotlib.use("Agg")

# None means "keep every dimension", i.e. the uncompressed baseline.
DEFAULT_COMPONENTS = [2, 3, 5, 10, 20, None]

# The uncompressed arm is the one the RAM wall excludes on the full partition:
# it is the reason run_knn is skipped there in the first place, so attempting it
# here would only reproduce that failure at the cost of a very long run.
BIG_DATA_COMPONENTS = [2, 3, 5, 10, 20]

DEFAULT_NEIGHBOURS = 31


def run_pca_knn_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    components=None,
    big_data=False,
    output_dir="results_notebook",
):
    """
    Sweep the number of principal components and refit K-NN on each projection.

    Records, for every setting, the cumulative explained variance, macro F1 and
    recall on the minority class, and the wall-clock seconds that fit plus
    prediction took -- the accuracy cost and the compute saving have to be read
    together for the trade-off to mean anything.

    Returns the sweep as a DataFrame.
    """
    if components is None:
        components = BIG_DATA_COMPONENTS if big_data else DEFAULT_COMPONENTS
    suffix = "_UsedBigData" if big_data else ""

    n_neighbors = _tuned_neighbours(big_data)
    print_log(
        f"PCA + K-NN sweep (K={n_neighbors}): does compressing the space rescue K-NN?"
    )

    rows = []
    for n in components:
        row = _one_setting(X_train, y_train, X_test, y_test, n, n_neighbors)
        rows.append(row)
        print(
            f"  components={str(row['n_components']):>4s}  "
            f"variance={row['Explained_variance']:6.1f}%  "
            f"macroF1={row['Macro_F1']:.4f}  "
            f"High recall={row['High_recall']:.4f}  "
            f"{row['Seconds']:6.1f}s"
        )

    if big_data:
        # Recorded rather than silently omitted: the absence is itself the result
        # the report claims in Section 2.2.
        rows.append({
            "n_components": "all (38)",
            "Explained_variance": 100.0,
            "Macro_F1": float("nan"),
            "High_recall": float("nan"),
            "Seconds": float("nan"),
            "Note": "not run: the uncompressed K-NN is what the memory wall excludes on this arena",
        })
        print("  components= all  not run on the full partition (memory wall, see Section 2.2)")

    sweep = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"pca_knn_sweep{suffix}.csv")
    sweep.to_csv(path, index=False)
    print(f"PCA + K-NN sweep saved in: {path}")

    _plot_sweep(sweep, output_dir, suffix, big_data)
    return sweep


def _tuned_neighbours(big_data):
    """Take K from the tuned model when one exists, so the setting cannot drift."""
    if model_exists("knn", use_big_data=big_data):
        return load_model("knn", use_big_data=big_data).get_params()["n_neighbors"]
    if model_exists("knn", use_big_data=False):
        # The full-data arena never tunes a K-NN; borrowing the balanced arena's K
        # keeps the two sweeps comparable instead of inventing a value.
        return load_model("knn", use_big_data=False).get_params()["n_neighbors"]
    return DEFAULT_NEIGHBOURS


def _one_setting(X_train, y_train, X_test, y_test, n, n_neighbors):
    """Project to n components (or keep the full space when n is None) and score."""
    if n is None:
        A, B, variance = X_train, X_test, 100.0
        label = f"all ({X_train.shape[1]})"
    else:
        pca = PCA(n_components=n, random_state=42)
        A = pca.fit_transform(X_train)
        B = pca.transform(X_test)
        variance = float(pca.explained_variance_ratio_.sum() * 100)
        label = n

    started = time.time()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(A, y_train)
    y_pred = knn.predict(B)
    seconds = time.time() - started

    return {
        "n_components": label,
        "Explained_variance": round(variance, 2),
        "Macro_F1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "High_recall": round(
            recall_score(y_test, y_pred, labels=[2], average="macro", zero_division=0), 4
        ),
        "Seconds": round(seconds, 1),
        "Note": "",
    }


def _plot_sweep(sweep, output_dir, suffix, big_data):
    """Macro F1 against components, with explained variance on a second axis."""
    plotted = sweep.dropna(subset=["Macro_F1"])
    labels = plotted["n_components"].astype(str).tolist()
    scores = plotted["Macro_F1"].tolist()
    variance = plotted["Explained_variance"].tolist()
    positions = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(positions, scores, marker="o", color="#12707C", lw=2, label="Macro F1")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Principal components retained")
    ax.set_ylabel("Macro F1-Score (common test set)")
    ax.grid(True, linestyle="--", alpha=0.5)

    twin = ax.twinx()
    twin.plot(positions, variance, marker="s", color="#A8642A", lw=1.5,
              linestyle=":", label="Explained variance")
    twin.set_ylabel("Cumulative explained variance (%)")
    twin.set_ylim(0, 105)

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels_ = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels_, loc="lower right")

    arena = "Entire dataset" if big_data else "Balanced subsample"
    ax.set_title(f"Compressing the space does not rescue K-NN ({arena})", fontsize=12, pad=15)
    fig.tight_layout()

    folder = os.path.join(output_dir, "KNN")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"pca_knn_sweep{suffix}.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"PCA + K-NN sweep plot saved in: {filepath}")
