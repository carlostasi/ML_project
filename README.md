# Predicting Irrigation Need

Multiclass classification of plot-level irrigation demand (`Low` / `Medium` /
`High`) from soil, climate and management attributes. Machine Learning course
project, MSc in Artificial Intelligence, University of Verona, A.Y. 2025/2026.

Author: Carlo Stasi (VR543606).

The written report is **`report.pdf`** in the repository root. Every number in it
is produced by the code here and traceable to a file under `results_notebook/`;
the map is in *Where the numbers come from* below.

## The problem

630,000 records from the Kaggle Playground Series (https://www.kaggle.com/competitions/playground-series-s6e4 [S6E4]), 19 predictive
attributes, three classes at 58.72% / 37.95% / **3.33%**. The rare `High` class
is the one that matters: failing to detect a drought costs a crop, while
irrigating a field that did not need it costs very little. Every design decision
in this project follows from that asymmetry, so the headline metric is macro F1
read **alongside** recall on `High`, never accuracy.

## Results at a glance

Trained on the full 504,000-row training partition with cost-sensitive
weighting, scored on the common 126,000-row test set at its natural class
distribution:

| Model | Macro F1 | Balanced acc. | Recall on `High` | On disk |
|---|---|---|---|---|
| Majority class *(baseline)* | 0.2466 | 0.3333 | 0.000 | — |
| Decision tree, depth 8 *(baseline)* | **0.9682** | 0.9595 | 0.907 | 39 KB |
| Random Forest, 300 trees | **0.9682** | 0.9601 | 0.909 | 494 MB |
| XGBoost | 0.9560 | **0.9671** | **0.941** | 1.03 MB |
| Linear SVM | 0.7536 | 0.7156 | 0.370 | 2 KB |
| K-NN, K = 11 | 0.7514 | 0.7071 | 0.376 | 157 MB |

The un-tuned decision tree ties the tuned Random Forest to four decimal places
at roughly one twelve-thousandth of its size. A paired bootstrap over 10,000
resamples of the test set puts that gap at −0.0001 with a 95% interval of
[−0.0011, +0.0009] — the only interval in the study that contains zero. The
RBF SVM is trained in the balanced arena only (0.8403 macro F1, 0.937 recall on
`High`); the tuning cost that excludes it from the full partition was measured
rather than assumed, and the measurement is in §2.2 of the report.

The two aggregate metrics disagree, which is one of the project's findings
rather than an accident: macro F1 puts the tree and the forest first, balanced
accuracy puts XGBoost first, because inverse-frequency weighting buys XGBoost
recall on `High` at the price of precision and only one of the two metrics
charges for that.

That disagreement was then checked from the outside. On the competition's
270,000 unseen rows the tree and the forest return the same label 99.84% of the
time, and XGBoost — predicted in advance to lead on balanced accuracy while
trailing on macro F1 — scored 0.96785 and 0.96630 on the two leaderboard halves,
within 0.0008 of the internal measurement on each.

## Running it

With conda:

```
conda env create -f environment.yml
conda activate irrigation_exam
python main.py
```

With pip instead:

```
pip install -r requirements.txt
python main.py
```

Both install the same nine libraries at the same versions, on Python 3.11. Those
versions are the ones the reported results were produced with. They are pinned
because several figures in the report are wall-clock timings, which move with the
BLAS build and the thread count; if a pin is unavailable on your platform,
loosening that one line will still run the pipeline.

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

`data/` and `saved_models*/` are both gitignored, so a fresh clone starts with
neither: download the CSVs first (see *Data* below) and leave `FORCE_RETRAIN` at
`True` for the first run of each arena. Afterwards the pickles are on disk and
`False` reuses them.

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
tools/make_submission.py    writes the Kaggle submissions and the agreement CSV
results_notebook/           CSVs and figures, one subfolder per model
report.pdf                  the written report
```

Models compared: K-NN, SVM (RBF and linear), Random Forest, XGBoost, each read
against two deliberately un-tuned baselines — a majority-class classifier and a
single decision tree.

## Where the numbers come from

No result in the report is typed in by hand. Each part of it reads from a file
the pipeline writes:

| Report | Artifact |
|---|---|
| §5.4 hyperparameters | `tables/hyperparams.tex`, generated from the pickles by `tools/dump_hyperparams.py` |
| §6.3 Model Comparison | `results_notebook/per_class_metrics*.csv`, `model_comparison*.csv` |
| §6.3, §6.8 confidence intervals | `results_notebook/model_bootstrap*.csv` |
| §6.4 Macro F1 vs the operational metric | `results_notebook/roc_auc*.csv` |
| §6.5 Choosing the Decision Rule by Cost | `results_notebook/cost_threshold*.csv` |
| §6.6 The Target Is a Threshold Rule | `results_notebook/depth_sweep*.csv` |
| §6.7 Dimensionality Reduction | `results_notebook/pca_knn_sweep*.csv` |
| §6.8 external validation | `results_notebook/kaggle_leaderboard.csv`, `submission_agreement.csv` |
| §6.9 Feature Importance and Ablation | `results_notebook/feature_ablation*.csv` |

A `*` marks a file that exists once per arena, the full-data copy carrying the
`_UsedBigData` suffix; figures follow the same convention under
`results_notebook/<Model>/`. `kaggle_leaderboard.csv` is the one file here
transcribed from outside rather than written by the pipeline, and it carries a
`Source` column saying so.

## What the project found

A single decision tree matches a tuned 300-tree Random Forest to four decimal
places while being four orders of magnitude smaller, the depth sweep peaks and
then decays, and the splits land on round agronomic values (`Soil_Moisture` 25,
`Temperature_C` 30, `Wind_Speed_kmh` 10). The synthetic target follows a
near-deterministic threshold rule, which explains the whole ranking of the
models: axis-aligned trees recover the rule, distance- and margin-based
estimators can only approximate it. `report.pdf` develops the argument.

## Data

`data/` is not versioned. Download `train.csv` and `test.csv` from the
competition page (https://www.kaggle.com/competitions/playground-series-s6e4) and place them there.
