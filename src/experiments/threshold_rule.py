import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score

from src.utils import print_log

matplotlib.use("Agg")

# None is a legitimate depth here (unbounded), so it cannot also mean "not given".
_UNSET = object()

# None is kept last so it reads as "unbounded" at the right-hand end of the sweep.
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, None]


def select_tree_depth(X_train, y_train, depths=None, cv=3, random_state=42):
    """
    Choose the reference tree's depth by cross-validation on the training data.

    The sweep below scores every depth on the test set. For a diagnostic that is
    the right thing to do -- the shape of the curve is what shows the target has
    fixed complexity -- but it is the wrong way to pick a model, and the tree
    reported in the results tables is a model. Its depth therefore comes from
    here, where no test row is touched, and the sweep keeps its separate role as
    evidence about the target rather than as a selection procedure.

    Returns (selected_depth, DataFrame of cross-validated scores).
    """
    depths = DEFAULT_DEPTHS if depths is None else depths

    print_log(f"Selecting the reference tree's depth by {cv}-fold CV on the training partition")

    rows = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        scores = cross_val_score(tree, X_train, y_train, cv=cv, scoring="f1_macro")
        rows.append(
            {
                "max_depth": "None" if depth is None else depth,
                "CV_Macro_F1": round(float(scores.mean()), 4),
                "CV_std": round(float(scores.std()), 4),
            }
        )
        print(f"  depth={str(depth):>4s}  CV macro F1={scores.mean():.4f} "
              f"(sd {scores.std():.4f})")

    table = pd.DataFrame(rows)
    best_label = table.loc[table["CV_Macro_F1"].idxmax(), "max_depth"]
    selected = None if best_label == "None" else int(best_label)
    print_log(f"Reference tree depth selected by cross-validation: {best_label}")

    return selected, table


def run_threshold_rule_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    transformer=None,
    depths=None,
    report_depth=3,
    output_dir="results_notebook",
    big_data=False,
    cv_table=None,
    selected_depth=_UNSET,
):
    """
    Sweep decision-tree depth, then report the shallow tree's split thresholds.

    X_train / X_test are the already-encoded matrices, so the sweep runs on the
    exact feature space the tuned models see. `transformer` is the fitted
    ColumnTransformer: it is optional, and is used only to invert the
    StandardScaler so that the printed thresholds are in agronomic units
    (mm, degrees C, ...) rather than in standard deviations.

    `cv_table` and `selected_depth` come from select_tree_depth. When present the
    cross-validated curve is written alongside the test curve and the selected
    depth is marked on the plot, which keeps the distinction visible: one curve
    selects, the other describes.

    Returns the sweep as a DataFrame.
    """
    depths = DEFAULT_DEPTHS if depths is None else depths
    suffix = "_UsedBigData" if big_data else ""

    print_log("Depth sweep: how much tree depth does this target actually need?")

    rows = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        macro_f1 = f1_score(y_test, tree.predict(X_test), average="macro")
        rows.append(
            {
                "max_depth": "None" if depth is None else depth,
                "n_leaves": int(tree.get_n_leaves()),
                "Macro_F1": round(macro_f1, 4),
            }
        )
        print(f"  depth={str(depth):>4s}  leaves={tree.get_n_leaves():>6d}  macro F1={macro_f1:.4f}")

    sweep = pd.DataFrame(rows)

    if cv_table is not None:
        sweep = sweep.merge(cv_table, on="max_depth", how="left")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"depth_sweep{suffix}.csv")
    sweep.to_csv(csv_path, index=False)
    print(f"Depth sweep saved in: {csv_path}")

    _plot_depth_sweep(sweep, output_dir, suffix, big_data, selected_depth)
    _print_thresholds(X_train, y_train, transformer, report_depth)

    return sweep


def _plot_depth_sweep(sweep, output_dir, suffix, big_data, selected_depth=_UNSET):
    """Plot macro F1 against depth: the test curve, and the CV curve that selects."""
    labels = sweep["max_depth"].astype(str).tolist()
    scores = sweep["Macro_F1"].tolist()
    positions = range(len(labels))

    plt.figure(figsize=(9, 5.5))
    plt.plot(positions, scores, marker="o", color="teal", lw=2,
             label="Macro F1 (common test set)")

    if "CV_Macro_F1" in sweep.columns and sweep["CV_Macro_F1"].notna().any():
        plt.plot(positions, sweep["CV_Macro_F1"].tolist(), marker="s", lw=1.5,
                 color="#A8642A", linestyle=":",
                 label="Macro F1 (3-fold CV on training)")

    mark_label = None
    if selected_depth is not _UNSET:
        candidate = "None" if selected_depth is None else str(selected_depth)
        if candidate in labels:
            mark_label = candidate

    if mark_label is not None:
        idx = labels.index(mark_label)
        note = f"depth selected by CV: {mark_label} (test F1 = {scores[idx]:.4f})"
    else:
        idx = int(np.argmax(scores))
        note = f"peak: depth {labels[idx]} (F1 = {scores[idx]:.4f})"
    plt.scatter([idx], [scores[idx]], s=160, facecolors="none",
                edgecolors="crimson", lw=2, zorder=5, label=note)

    plt.xticks(list(positions), labels)
    plt.xlabel("Tree max_depth")
    plt.ylabel("Macro F1-Score")
    arena = "Entire dataset" if big_data else "Balanced subsample"
    plt.title(f"A single decision tree is enough - depth sweep ({arena})", fontsize=12, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    folder = os.path.join(output_dir, "Baselines")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"depth_sweep{suffix}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Depth sweep plot saved in: {filepath}")


def _print_thresholds(X_train, y_train, transformer, report_depth):
    """
    Print a shallow tree with its numeric thresholds mapped back to original
    units, so the round numbers the generator used are visible.
    """
    tree = DecisionTreeClassifier(max_depth=report_depth, random_state=42)
    tree.fit(X_train, y_train)

    if transformer is None:
        print(export_text(tree, decimals=4))
        return

    feature_names = list(transformer.get_feature_names_out())
    scaler = transformer.named_transformers_["num"]
    # The numeric block is the first one in the ColumnTransformer, so the i-th
    # scaled column corresponds to the i-th entry of the scaler's statistics.
    scaled_cols = {f"num__{name}": i for i, name in enumerate(scaler.feature_names_in_)}

    print_log(f"Split thresholds of a depth-{report_depth} tree, in original units:")
    for line in export_text(tree, feature_names=feature_names, decimals=4).splitlines():
        for column, i in scaled_cols.items():
            if column not in line:
                continue
            # export_text writes the same split as "<= t" on the left branch and
            # "> t" on the right, so both operators have to be rescaled or the
            # right-hand branches would keep showing standard deviations.
            operator = "<=" if "<=" in line else (">" if ">" in line else None)
            if operator is None:
                continue
            head, _, threshold = line.partition(operator)
            real = float(threshold) * scaler.scale_[i] + scaler.mean_[i]
            line = f"{head}{operator} {real:.2f}  [{column.replace('num__', '')}, original units]"
            break
        print(line)
