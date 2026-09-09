# Predicting Irrigation Need

Multiclass classification of plot-level irrigation demand (`Low` / `Medium` /
`High`) from soil, climate and management attributes. Machine Learning course
project, MSc in Artificial Intelligence, University of Verona, A.Y. 2025/2026.

Author: Carlo Stasi (VR543606).

## The problem

630,000 records from the Kaggle Playground Series (S6E4), 19 predictive
attributes, three classes at 58.72% / 37.95% / **3.33%**. The rare `High` class
is the one that matters: failing to detect a drought costs a crop, while
irrigating a field that did not need it costs very little. Every design decision
in this project follows from that asymmetry, so the headline metric is macro F1
read **alongside** recall on `High`, never accuracy.

## Running it

```
pip install -r requirements.txt
python main.py
```

`main.py` is the only entry point and runs the whole pipeline: load →
feature engineering → stratified split → (optional) balancing → preprocessing →
baselines → tuning of each model → evaluation → figures → four experiments →
result CSVs.

Three switches at the top of `main()` control what it does:

| Flag | Effect |
|---|---|
| `USE_BIG_DATA` | `False`: train on a class-balanced subsample of the *training* partition (16,807 rows per class). `True`: train on all 504,000 training rows with cost-sensitive weighting. The test set is the same 126,000 rows either way. |
| `FORCE_RETRAIN` | `False`: reuse the models cached in `saved_models*/`. `True`: retune everything from scratch (hours on the full arena). |
| `SAMPLE_PER_CLASS` | Upper bound per class in the balanced arena. Not binding: the rarest class in the training split is smaller. |

Expect roughly 30 minutes for a cached run of the balanced arena and 1–1.5 hours
for the full one, dominated by the feature ablation and the PCA sweep rather
than by model fitting.

## How the evaluation is set up

The data is split **once**, on its natural class distribution, into 504,000
training and 126,000 test rows. Balancing, when enabled, touches the training
partition only. Both arenas are therefore scored on the same representative test
set and their numbers are directly comparable — an earlier version of this code
balanced before splitting, which made every reported score a measurement on a
population that does not exist in the field.

## Layout

```
main.py                     the pipeline, phase by phase
src/data/                   loading, feature engineering, balancing, preprocessing
src/models/                 one run_<model>() per algorithm, plus persistence
src/metrics/                evaluation and plots
src/experiments/            threshold rule, feature ablation, PCA+K-NN, cost threshold
tools/dump_hyperparams.py   regenerates tables/hyperparams.tex from the pickles
results_notebook/           CSVs and figures, one subfolder per model
report.tex                  the report (compiled on Overleaf)
```

Models compared: K-NN, SVM (RBF and linear), Random Forest, XGBoost, each read
against two deliberately un-tuned baselines — a majority-class classifier and a
single decision tree.

## What the project found

A single decision tree matches a tuned 300-tree Random Forest to four decimal
places while being four orders of magnitude smaller, the depth sweep peaks and
then decays, and the splits land on round agronomic values (`Soil_Moisture` 25,
`Temperature_C` 30, `Wind_Speed_kmh` 10). The synthetic target follows a
near-deterministic threshold rule, which explains the whole ranking of the
models: axis-aligned trees recover the rule, distance- and margin-based
estimators can only approximate it. `report.tex` develops the argument.

## Data

`data/` is not versioned. Download `train.csv` and `test.csv` from the
competition page and place them there.
