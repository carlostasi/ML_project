"""
Paired bootstrap over the test set.

Two estimators scored on the same rows share the luck of the draw, so comparing
them by resampling both on the *same* resampled rows cancels that shared
component and leaves an interval for the difference alone. This module was
written for the feature ablation, where the arms differ in their inputs, and is
used unchanged for the model comparison, where they differ in the estimator: in
both cases the object of study is a difference in macro F1 between two sets of
predictions on one test set.

The intervals cover test-sample uncertainty only. The models are held fixed, so
nothing here says what would happen if they were retrained on different data.
"""

import os

import numpy as np
import pandas as pd

from src.utils import print_log


def run_paired_bootstrap(y_true, predictions, n_resamples=10000, suffix="",
                         output_dir="results_notebook", random_state=42,
                         filename="feature_ablation_bootstrap", reference=None,
                         label_width=30):
    """
    Compare every arm in `predictions` against one reference arm.

    `predictions` maps a label to an array of predicted classes. `reference`
    names the arm everything else is measured against; without it the first key
    is used, which is what the feature ablation wants (its first arm is the full
    feature set). The model comparison passes the best-scoring model instead, so
    every reported delta reads as "how far behind the winner".

    Returns the table as a DataFrame, or None when there is nothing to compare.
    """
    labels = list(predictions)
    if len(labels) < 2 or y_true is None:
        return None

    reference_label = labels[0] if reference is None else reference
    if reference_label not in predictions:
        raise ValueError(f"reference {reference_label!r} is not among the arms")

    rng = np.random.default_rng(random_state)
    truth = np.asarray(y_true, dtype=np.int8)
    reference_pred = np.asarray(predictions[reference_label], dtype=np.int8)
    n = truth.shape[0]

    print_log(
        f"Paired bootstrap ({n_resamples} resamples), reference arm: {reference_label}"
    )

    rows = []
    for label in labels:
        if label == reference_label:
            continue
        arm = np.asarray(predictions[label], dtype=np.int8)

        # One row -> one of 27 cells, indexed as 9*true + 3*reference + arm.
        cells = np.bincount(9 * truth + 3 * reference_pred + arm, minlength=27)
        draws = rng.multinomial(n, cells / n, size=n_resamples).reshape(-1, 3, 3, 3)

        f1_reference = _macro_f1_from_cm(draws.sum(axis=3))   # sum out the arm axis
        f1_arm = _macro_f1_from_cm(draws.sum(axis=2))         # sum out the reference axis
        deltas = f1_arm - f1_reference

        observed = (_macro_f1_from_cm(cells.reshape(1, 3, 3, 3).sum(axis=2))
                    - _macro_f1_from_cm(cells.reshape(1, 3, 3, 3).sum(axis=3)))[0]
        low, high = np.percentile(deltas, [2.5, 97.5])
        # Share of resamples in which the arm did not come out behind the reference.
        share_no_harm = float((deltas >= 0).mean())

        rows.append({
            "Arm": label,
            "Reference": reference_label,
            "Observed_delta": round(float(observed), 4),
            "CI95_low": round(float(low), 4),
            "CI95_high": round(float(high), 4),
            "Share_no_harm": round(share_no_harm, 4),
            "Excludes_zero": "yes" if low > 0 or high < 0 else "no",
        })
        print(f"  {label:{label_width}s} delta={observed:+.4f}  "
              f"95% CI [{low:+.4f}, {high:+.4f}]  "
              f"{'distinguishable from zero' if (low > 0 or high < 0) else 'NOT distinguishable from zero'}")

    table = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{filename}{suffix}.csv")
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
