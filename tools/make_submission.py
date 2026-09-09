"""
Build Kaggle submission files from the models this project reports.

The 270,000 rows of `data/test.csv` are the only genuinely external check available:
they come from the same generator but their labels are not ours to see, so a score on
them cannot be an artefact of the split we chose or of anything we tuned. Everything
in the report rests on one 126,000-row test set that we carved out ourselves; this
puts the same models in front of twice as many rows nobody has looked at.

    python tools/make_submission.py

Writes one CSV per model into submissions/, in the `id,Irrigation_Need` format of
`data/sample_submission.csv`.

Two models are exported on purpose. The central claim of the report is that the
depth-8 decision tree and the 300-tree Random Forest are statistically
indistinguishable on the held-out set while differing by four orders of magnitude in
size. Submitting both puts that claim in front of an independent sample: if the two
leaderboard scores land on top of each other, the claim survives a test it was never
designed for.

The models are used exactly as the report describes them -- fitted on the 504,000-row
training partition, not refitted on all 630,000 labelled rows. A refit would score
better and would no longer be the model the report is about.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd  # noqa: E402

from src.data.preprocessing import (  # noqa: E402
    dataset_setup,
    features_engineering,
    get_pipeline_transformer,
    load_data,
)
from src.models import load_model, model_exists  # noqa: E402
from src.utils import print_log  # noqa: E402

TRAIN_PATH = os.path.join("data", "train.csv")
TEST_PATH = os.path.join("data", "test.csv")
OUTPUT_DIR = "submissions"

# dataset_setup maps the target as {Low: 0, Medium: 1, High: 2}; this inverts it.
LABELS = {0: "Low", 1: "Medium", 2: "High"}

# (key on disk, name for the filename and the log)
#
# The first two are the pair the report's central claim is about. XGBoost is here for
# a different reason: the competition scores balanced accuracy, the macro average of
# per-class recall, and on that metric XGBoost leads both of them (0.9671 against
# 0.9601 and 0.9595) even though macro F1 ranks it last of the three. Submitting it
# turns that into a prediction the leaderboard can check.
MODELS = [
    ("decision_tree", "decision_tree_d8"),
    ("random_forest", "random_forest"),
    ("xgboost", "xgboost"),
]


def main():
    for path in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(path):
            print_log(f"Error: {path} not found")
            return

    # The transformer has to be the one the models were fitted through, so it is
    # refitted here on the same 504,000 rows: same split, same seed, same columns.
    print_log("Refitting the transformer on the full-data training partition")
    X, y = load_data(TRAIN_PATH)
    X_train, _, y_train, _ = dataset_setup(X, y)
    transformer = get_pipeline_transformer()
    transformer.fit(X_train)
    print(f"  training rows: {X_train.shape[0]}")

    print_log("Preparing the Kaggle test set")
    test = pd.read_csv(TEST_PATH)
    ids = test["id"].copy()
    # features_engineering only adds columns, and the ColumnTransformer selects the
    # ones it needs by name, so `id` can ride along without being dropped first.
    test = features_engineering(test)
    X_kaggle = transformer.transform(test)
    print(f"  rows: {X_kaggle.shape[0]}, encoded features: {X_kaggle.shape[1]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predicted = {}
    for key, name in MODELS:
        if not model_exists(key, use_big_data=True):
            print_log(f"[INFO] {key} not found in saved_models_BigData, skipped.")
            continue

        model = load_model(key, use_big_data=True)
        predictions = pd.Series(model.predict(X_kaggle)).map(LABELS)
        predicted[name] = predictions

        submission = pd.DataFrame({"id": ids, "Irrigation_Need": predictions})
        path = os.path.join(OUTPUT_DIR, f"submission_{name}.csv")
        submission.to_csv(path, index=False)

        # Sanity check rather than a result: the predicted shares should sit near the
        # training prior (58.72 / 37.95 / 3.33). A wild departure would mean the
        # encoding went wrong somewhere between the two files.
        shares = (submission["Irrigation_Need"].value_counts(normalize=True) * 100).round(2)
        print_log(f"{name}: {path}")
        for label in ("Low", "Medium", "High"):
            print(f"  {label:<7} {shares.get(label, 0.0):>6.2f}%")

    if len(predicted) >= 2:
        _record_agreement(predicted)

    print_log("Done. Upload the files in submissions/ to the competition page.")


def _record_agreement(predicted):
    """
    How often do the two models return the same label on rows neither has seen?

    This needs no labels, so it is a complete result on its own and does not wait on
    a leaderboard. The paired bootstrap of the report says the two models score
    alike; this says something stronger, that they largely *decide* alike, and it
    says it on 270,000 rows against the 126,000 the report is otherwise built on.
    """
    (name_a, a), (name_b, b) = list(predicted.items())[:2]
    agree = (a == b)

    # Low and High are two steps apart on an ordered scale, so a disagreement between
    # them is a different kind of error from one between neighbours.
    extremes = ((a == "Low") & (b == "High")) | ((a == "High") & (b == "Low"))

    row = {
        "Model_a": name_a,
        "Model_b": name_b,
        "Rows": len(a),
        "Identical": int(agree.sum()),
        "Identical_pct": round(100 * float(agree.mean()), 3),
        "Disagreements": int((~agree).sum()),
        "Adjacent": int(((~agree) & (~extremes)).sum()),
        "Low_vs_High": int(extremes.sum()),
    }

    print_log(f"Agreement between {name_a} and {name_b} on the Kaggle test set")
    print(f"  identical predictions : {row['Identical']} of {row['Rows']} "
          f"({row['Identical_pct']}%)")
    print(f"  disagreements         : {row['Disagreements']} "
          f"({row['Adjacent']} between neighbouring classes, "
          f"{row['Low_vs_High']} between Low and High)")

    path = os.path.join("results_notebook", "submission_agreement.csv")
    os.makedirs("results_notebook", exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Agreement saved in: {path}")


if __name__ == "__main__":
    main()
