"""
Choosing the decision rule by expected cost instead of by arg-max.

Every score elsewhere in this project comes from arg-max over the predicted
class probabilities, which is the rule that minimises expected error when all
errors cost the same. This project assumes they do not: a missed drought ruins a
crop, a needless irrigation costs water. The right rule under that assumption is
to predict `High` whenever the expected cost of not doing so exceeds the
expected cost of doing so, which reduces to a threshold on P(High).

The threshold is chosen on a validation slice carved out of the *training*
partition, never on the test set. A threshold tuned on the test set would report
its own selection back as a result, which is the same mistake the depth sweep
avoids by selecting its depth with cross-validation.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

from src.models import load_model, model_exists
from src.utils import print_log

matplotlib.use("Agg")

HIGH = 2
DEFAULT_LAMBDAS = (1, 2, 5, 10, 20, 50)
TUNED_KEYS = ("n_estimators", "max_depth", "learning_rate", "subsample")


def run_cost_threshold_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    big_data=False,
    lambdas=DEFAULT_LAMBDAS,
    output_dir="results_notebook",
    random_state=42,
):
    """
    Sweep the decision threshold on P(High) and pick one per cost ratio.

    lambda is the price of a missed drought expressed in needless irrigations:
    lambda = 10 says one undetected `High` field costs as much as ten fields
    watered that did not need it. lambda = 1 recovers a symmetric criterion and
    is reported as the control.

    Returns the results table as a DataFrame, or None when no tuned XGBoost
    exists for this arena.
    """
    suffix = "_UsedBigData" if big_data else ""

    if not model_exists("xgboost", use_big_data=big_data):
        print_log("[INFO] Cost-threshold experiment skipped: no tuned XGBoost to read.")
        return None

    print_log("Cost-threshold experiment: is arg-max the rule this project wants?")

    tuned = load_model("xgboost", use_big_data=big_data)
    params = {key: tuned.get_params()[key] for key in TUNED_KEYS}
    print(f"  XGBoost settings held fixed at the tuned values: {params}")

    # The tuned model has seen every training row, so its probabilities on those
    # rows are optimistic and cannot select a threshold. A twin fitted on 80% of
    # the training partition supplies honest probabilities on the held-out 20%.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    twin = XGBClassifier(random_state=random_state, eval_metric="mlogloss", **params)
    twin.fit(X_fit, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))

    y_val = np.asarray(y_val)
    p_val = twin.predict_proba(X_val)[:, HIGH]
    other_val = np.argmax(twin.predict_proba(X_val)[:, :HIGH], axis=1)

    thresholds = np.unique(
        np.concatenate([
            np.linspace(0.01, 0.99, 99),
            np.quantile(p_val, np.linspace(0.001, 0.999, 200)),
        ])
    )

    y_test = np.asarray(y_test)
    p_test = tuned.predict_proba(X_test)[:, HIGH]
    other_test = np.argmax(tuned.predict_proba(X_test)[:, :HIGH], axis=1)

    rows = [_argmax_row(tuned, X_test, y_test)]
    chosen = {}
    for lam in lambdas:
        costs = np.array([_cost(y_val, _rule(p_val, other_val, t), lam) for t in thresholds])
        best = thresholds[int(np.argmin(costs))]
        chosen[lam] = (best, costs)
        rows.append(
            _score_row(
                label=f"expected cost, lambda = {lam}",
                threshold=best,
                y_true=y_test,
                y_pred=_rule(p_test, other_test, best),
                lam=lam,
                val_cost=float(costs.min()) / len(y_val) * 1000,
            )
        )
        print(f"  lambda={lam:>3}  threshold={best:.4f}  "
              f"recall(High)={rows[-1]['High_recall']:.4f}  "
              f"precision(High)={rows[-1]['High_precision']:.4f}  "
              f"macro F1={rows[-1]['Macro_F1']:.4f}")

    table = pd.DataFrame(rows)
    brier = brier_score_loss((y_test == HIGH).astype(int), p_test)
    table["Brier_High"] = round(float(brier), 5)
    print(f"  Brier score for P(High) on the test set: {brier:.5f}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"cost_threshold{suffix}.csv")
    table.to_csv(path, index=False, float_format="%.4f")
    print(f"Cost-threshold table saved in: {path}")

    _plot(p_test, y_test, table, chosen, thresholds, len(y_val), output_dir, suffix, big_data)
    return table


def _rule(p_high, other, threshold):
    """Predict High when P(High) clears the threshold, else the better of Low/Medium."""
    return np.where(p_high >= threshold, HIGH, other)


def _cost(y_true, y_pred, lam):
    """lam missed droughts against one needless irrigation, counted in fields."""
    missed = int(np.sum((y_true == HIGH) & (y_pred != HIGH)))
    false_alarm = int(np.sum((y_true != HIGH) & (y_pred == HIGH)))
    return lam * missed + false_alarm


def _score_row(label, threshold, y_true, y_pred, lam, val_cost):
    tp = int(np.sum((y_true == HIGH) & (y_pred == HIGH)))
    predicted = int(np.sum(y_pred == HIGH))
    actual = int(np.sum(y_true == HIGH))
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Rule": label,
        "Lambda": lam,
        "Threshold": round(float(threshold), 4) if threshold is not None else "",
        "High_precision": round(precision, 4),
        "High_recall": round(recall, 4),
        "High_f1": round(f1, 4),
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Predicted_High": predicted,
        "Val_cost_per_1000": round(val_cost, 2) if val_cost is not None else "",
        "Test_cost_per_1000": round(_cost(y_true, y_pred, lam) / len(y_true) * 1000, 2)
        if lam is not None else "",
    }


def _argmax_row(model, X_test, y_test):
    row = _score_row(
        label="arg-max (the rule used everywhere else)",
        threshold=None,
        y_true=y_test,
        y_pred=np.asarray(model.predict(X_test)),
        lam=None,
        val_cost=None,
    )
    row["Lambda"] = ""
    return row


def _plot(p_test, y_test, table, chosen, thresholds, n_val, output_dir, suffix, big_data):
    """Left: where the operating points sit on the test PR curve. Right: the cost curves."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))

    precision, recall, _ = precision_recall_curve((y_test == HIGH).astype(int), p_test)
    left.plot(recall, precision, color="#12707C", lw=2, label="P/R frontier for High")

    marks = table[table["Rule"] != "arg-max (the rule used everywhere else)"]
    left.scatter(marks["High_recall"], marks["High_precision"], s=55, zorder=5,
                 color="#A8642A", label="chosen by expected cost")
    for _, r in marks.iterrows():
        left.annotate(f"$\\lambda$={r['Lambda']}", (r["High_recall"], r["High_precision"]),
                      textcoords="offset points", xytext=(6, 5), fontsize=8)
    argmax = table[table["Rule"] == "arg-max (the rule used everywhere else)"].iloc[0]
    left.scatter([argmax["High_recall"]], [argmax["High_precision"]], s=110, zorder=6,
                 marker="*", color="crimson", label="arg-max")

    left.set_xlabel("Recall on High (test set)")
    left.set_ylabel("Precision on High (test set)")
    left.set_title("Moving the operating point", fontsize=11)
    left.grid(True, linestyle="--", alpha=0.5)
    left.legend(loc="lower left", fontsize=9)

    for lam in sorted(chosen)[:6]:
        best, costs = chosen[lam]
        right.plot(thresholds, costs / n_val * 1000, lw=1.5, label=f"$\\lambda$ = {lam}")
        right.axvline(best, color="grey", lw=0.6, linestyle=":")
    right.set_xlabel("Threshold on P(High), chosen on validation")
    right.set_ylabel("Expected cost per 1,000 fields")
    right.set_yscale("log")
    right.set_title("Where each cost ratio puts the threshold", fontsize=11)
    right.grid(True, linestyle="--", alpha=0.5)
    right.legend(fontsize=9)

    arena = "Entire dataset" if big_data else "Balanced subsample"
    fig.suptitle(f"Cost-sensitive decision rule, XGBoost ({arena})", fontsize=12)
    fig.tight_layout()

    folder = os.path.join(output_dir, "XGBoost")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"cost_threshold{suffix}.png")
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Cost-threshold plot saved in: {filepath}")
