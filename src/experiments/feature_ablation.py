import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

from src.data.preprocessing import (
    load_data,
    balance_training_set,
    dataset_setup,
    get_pipeline_transformer,
)
from src.models import load_model, model_exists
from src.utils import print_log

ENGINEERED = ["Thermal_impact", "Soil_Health", "Water_Deficit"]

# (label for the results table, columns to drop, whether to add the proxy)
ARMS = [
    ("All 38 features (reference)", [], False),
    ("No engineered features", ENGINEERED, False),
    ("No Thermal_impact", ["Thermal_impact"], False),
    ("No Soil_Health", ["Soil_Health"], False),
    ("No Water_Deficit", ["Water_Deficit"], False),
    ("Plus standalone E", [], True),
]


def run_feature_ablation(
    dataset_path,
    big_data=False,
    sample_per_class=21000,
    output_dir="results_notebook",
):
    """
    Retrain a fixed-hyperparameter XGBoost and depth-8 tree on each feature set.

    The data is reloaded here rather than taken as an argument, because the arms
    differ in which columns exist at all: Evaporation_proxy is produced by
    features_engineering only on request, so a matrix built upstream could not
    contain it. Everything else -- the split, the seed, the balancing -- matches
    the main pipeline exactly, so the numbers are comparable with the rest of
    the results.

    Returns the comparison as a DataFrame.
    """
    suffix = "_UsedBigData" if big_data else ""
    print_log("Feature ablation: do the engineered terms earn their place?")

    if not model_exists("xgboost", use_big_data=big_data):
        print_log("[INFO] Feature ablation skipped: no tuned XGBoost to take settings from.")
        return None

    tuned = load_model("xgboost", use_big_data=big_data)
    xgb_params = {
        key: tuned.get_params()[key]
        for key in ("n_estimators", "max_depth", "learning_rate", "subsample")
    }
    print(f"  XGBoost settings held fixed at the tuned values: {xgb_params}")

    # The linear arm takes its C from the tuned LinearSVC when one exists for
    # this arena; the balanced arena has no such pickle (it trains an RBF SVM
    # instead), so it falls back to the scikit-learn default and says so.
    if model_exists("linear_svm", use_big_data=big_data):
        svm_c = load_model("linear_svm", use_big_data=big_data).get_params()["C"]
        print(f"  Linear SVM C taken from the tuned model: {svm_c}")
    else:
        svm_c = 1.0
        print(f"  Linear SVM C defaulted to {svm_c} (no tuned LinearSVC in this arena)")

    rows = []
    linear_predictions = {}   # per-arm Linear SVM predictions, kept for the bootstrap
    truth = None
    for label, drop, add_proxy in ARMS:
        X, y = load_data(dataset_path, keep_evaporation_proxy=add_proxy)
        X_train, X_test, y_train, y_test = dataset_setup(X, y)
        truth = y_test
        if not big_data:
            X_train, y_train = balance_training_set(
                X_train, y_train, sample_size_per_class=sample_per_class
            )

        transformer = get_pipeline_transformer(
            drop_numeric=drop or None,
            add_numeric=["Evaporation_proxy"] if add_proxy else None,
        )
        A = transformer.fit_transform(X_train)
        B = transformer.transform(X_test)

        row = {"Feature_set": label, "N_features": A.shape[1]}
        for model, name, weighted in _arm_models(xgb_params, svm_c):
            if weighted:
                model.fit(A, y_train, sample_weight=compute_sample_weight("balanced", y_train))
            else:
                model.fit(A, y_train)
            y_pred = model.predict(B)
            if name == "LinearSVM":
                linear_predictions[label] = np.asarray(y_pred, dtype=np.int8)
            row[f"{name}_Macro_F1"] = round(f1_score(y_test, y_pred, average="macro"), 4)
            row[f"{name}_High_recall"] = round(
                recall_score(y_test, y_pred, labels=[2], average="macro", zero_division=0), 4
            )
        rows.append(row)
        print(
            f"  {label:30s} {row['N_features']:>3d} feat  "
            f"XGB F1={row['XGBoost_Macro_F1']:.4f}  "
            f"Tree F1={row['Tree_Macro_F1']:.4f}  "
            f"LinSVM F1={row['LinearSVM_Macro_F1']:.4f}"
        )

    table = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"feature_ablation{suffix}.csv")
    table.to_csv(path, index=False)
    print(f"Feature ablation saved in: {path}")

    run_paired_bootstrap(truth, linear_predictions, suffix=suffix, output_dir=output_dir)
    return table


def run_paired_bootstrap(y_true, predictions, n_resamples=10000, suffix="",
                         output_dir="results_notebook", random_state=42):
    labels = list(predictions)
    if len(labels) < 2 or y_true is None:
        return None

    rng = np.random.default_rng(random_state)
    truth = np.asarray(y_true, dtype=np.int8)
    reference_label = labels[0]
    reference = predictions[reference_label]
    n = truth.shape[0]

    print_log(f"Paired bootstrap on the Linear SVM arms ({n_resamples} resamples)")

    rows = []
    for label in labels[1:]:
        arm = predictions[label]

        # One row -> one of 27 cells, indexed as 9*true + 3*reference + arm.
        cells = np.bincount(9 * truth + 3 * reference + arm, minlength=27)
        draws = rng.multinomial(n, cells / n, size=n_resamples).reshape(-1, 3, 3, 3)

        f1_reference = _macro_f1_from_cm(draws.sum(axis=3))   # sum out the arm axis
        f1_arm = _macro_f1_from_cm(draws.sum(axis=2))         # sum out the reference axis
        deltas = f1_arm - f1_reference

        observed = (_macro_f1_from_cm(cells.reshape(1, 3, 3, 3).sum(axis=2))
                    - _macro_f1_from_cm(cells.reshape(1, 3, 3, 3).sum(axis=3)))[0]
        low, high = np.percentile(deltas, [2.5, 97.5])
        # Share of resamples in which removing the feature did not hurt at all.
        share_no_harm = float((deltas >= 0).mean())

        rows.append({
            "Feature_set": label,
            "Observed_delta": round(float(observed), 4),
            "CI95_low": round(float(low), 4),
            "CI95_high": round(float(high), 4),
            "Share_no_harm": round(share_no_harm, 4),
            "Excludes_zero": "yes" if low > 0 or high < 0 else "no",
        })
        print(f"  {label:30s} delta={observed:+.4f}  "
              f"95% CI [{low:+.4f}, {high:+.4f}]  "
              f"{'distinguishable from zero' if (low > 0 or high < 0) else 'NOT distinguishable from zero'}")

    table = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"feature_ablation_bootstrap{suffix}.csv")
    table.to_csv(path, index=False)
    print(f"Bootstrap saved in: {path}")
    return table


def _macro_f1_from_cm(cm):
    """
    Macro F1 from a stack of confusion matrices, shaped (samples, true, pred).

    Working from the confusion matrix rather than from label vectors is what
    makes the bootstrap cheap: the resampled matrices come straight out of the
    multinomial draw, and no per-row comparison is ever repeated.
    """
    cm = cm.astype(np.float64)
    tp = np.einsum("bcc->bc", cm)
    predicted = cm.sum(axis=1)
    actual = cm.sum(axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    return f1.mean(axis=1)


def _arm_models(xgb_params, svm_c):
    """
    The three estimators each arm is measured with, as (model, name, weighted).

    Linear SVM is the one that matters for the report's argument: it cannot form
    a product or a ratio of two columns on its own, so if the engineered terms
    carry real information this is where it has to show up.

    XGBoost carries the tuned settings and the same balanced sample weights the
    main pipeline uses; the depth-8 tree is the interpretable reference, kept
    un-weighted exactly as run_decision_tree leaves it. Both can rebuild an
    interaction by splitting repeatedly on its constituents, which is why they
    are expected to be indifferent to the engineering. The Random Forest is
    deliberately excluded: refitting a 300-tree, 494 MB forest six times would
    cost hours and add nothing these three do not already show.
    """
    return [
        (XGBClassifier(random_state=42, eval_metric="mlogloss", **xgb_params), "XGBoost", True),
        (DecisionTreeClassifier(max_depth=8, random_state=42), "Tree", False),
        (
            LinearSVC(C=svm_c, dual=False, class_weight="balanced",
                      random_state=42, max_iter=2000),
            "LinearSVM",
            False,
        ),
    ]
