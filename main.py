import os

# This has to run before numpy, scikit-learn or xgboost are imported. Each of
# them loads an OpenMP runtime, and Intel's aborts the process when it finds
# LLVM's already initialised. The line used to sit after the imports, where the
# runtimes are already in memory and setting the variable does nothing at all.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.preprocessing import (
    load_data,
    balance_training_set,
    get_pipeline_transformer,
    dataset_setup,
)
from src.models import (
    run_svm,
    run_knn,
    run_knn_fixed,
    tuned_k,
    run_xgboost,
    run_random_forest,
    run_majority_baseline,
    run_decision_tree,
    save_model,
    load_model,
    model_exists,
)
from src.experiments import (
    run_threshold_rule_experiment,
    select_tree_depth,
    run_feature_ablation,
    run_pca_knn_experiment,
    run_cost_threshold_experiment,
    run_paired_bootstrap,
)
from src.metrics import (
    evaluate_model,
    plot_save_confusion_matrix,
    plot_decision_boundaries_2d,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_multiclass_roc,
)
from src.utils import print_log
import pandas as pd


def main():
    USE_BIG_DATA = False
    FORCE_RETRAIN = False  # Set to True to ignore saved models and retrain from scratch
    SAMPLE_PER_CLASS = 21000  # Upper bound per class in the balanced arena

    dataset_path = os.path.join("data", "train.csv")

    if not os.path.exists(dataset_path):
        print_log(f"Error: Dataset file not found")
        return

    print_log("=== PHASE 1: Loading data & Feature Engineering ===")
    X, y = load_data(dataset_path)
    print_log(f"Dataset ready. Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    # print("Colonne effettive in X:", X.columns.tolist())

    print_log("=== PHASE 2: Split and Preprocessing ===")
    # Split BEFORE any balancing, so the test set keeps the natural class
    # distribution and is identical across the two arenas: whatever the training
    # regime, every model is scored on the same, representative test set.
    X_train, X_test, y_train, y_test = dataset_setup(X, y)
    test_shares = (y_test.value_counts(normalize=True).sort_index() * 100).round(2)
    print_log(
        f"Common test set: {X_test.shape[0]} rows "
        f"(Low {test_shares[0]}%, Medium {test_shares[1]}%, High {test_shares[2]}%)"
    )

    if USE_BIG_DATA:
        print_log(f"Arena: FULL DATA. Training rows: {X_train.shape[0]}")
    else:
        X_train, y_train = balance_training_set(
            X_train, y_train, sample_size_per_class=SAMPLE_PER_CLASS
        )
        per_class = int(y_train.value_counts().min())
        print_log(
            f"Arena: BALANCED. Training rows: {X_train.shape[0]} "
            f"({per_class} per class, capped by the rarest class in the training split)"
        )

    plot_correlation_matrix(X_train, big_data=USE_BIG_DATA)

    transformer = get_pipeline_transformer()
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)
    print_log(
        f"Pre-processing completed. Feature post-encoding: {X_train_processed.shape[1]}"
    )

    print_log("=== PHASE 3: Training and tuning models ===")

    # --- Baselines ---
    # Trained first and evaluated alongside the tuned models, so the headline
    # score is never reported without a floor (majority class) and an
    # interpretable reference (a single tree) next to it.
    print("\n" + "=" * 50)
    print("BASELINES")
    print("=" * 50)
    best_majority, _ = run_majority_baseline(X_train_processed, y_train)
    save_model(best_majority, "majority_baseline", use_big_data=USE_BIG_DATA)

    # The depth of the reference tree is chosen by cross-validation on the
    # training partition. Reading it off the depth sweep instead would mean
    # selecting a model on the test set the same model is then scored on.
    tree_depth, depth_cv = select_tree_depth(X_train_processed, y_train)
    best_tree, _ = run_decision_tree(X_train_processed, y_train, max_depth=tree_depth)
    save_model(best_tree, "decision_tree", use_big_data=USE_BIG_DATA)
    tree_label = f"Decision Tree (d={tree_depth})"

    fast_svm_mode = True if USE_BIG_DATA else False
    svm_model_name = "linear_svm" if fast_svm_mode else "rbf_svm"

    # --- K-NN ---
    print("\n" + "=" * 50)
    print("K-NN")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists("knn", use_big_data=USE_BIG_DATA) and not USE_BIG_DATA:
        best_knn = load_model("knn", use_big_data=USE_BIG_DATA)
    elif USE_BIG_DATA:
        # What the memory wall excludes on this arena is the grid search, not
        # the fit: ten values of K at cv=3 is thirty passes of the same pairwise
        # distance computation, each worker holding its own copy of the data.
        # One pass is affordable, so K-NN appears here as an untuned reference
        # at the K the balanced arena selected.
        #
        # It is deliberately not saved. Fitting a K-NN only stores the training
        # matrix, so the pickle would be ~150 MB of rows that already live in
        # data/train.csv, and caching saves nothing because the cost is all in
        # prediction. Worse, tools/dump_hyperparams.py reads saved_models*/ as
        # the record of what GridSearchCV selected, and a pickle here would make
        # the report's table claim a tuned K for an arena that never tuned one.
        borrowed = tuned_k(use_big_data=True)
        print_log(f"K-NN: no grid search on this arena, fitting once at K={borrowed}.")
        best_knn, _ = run_knn_fixed(X_train_processed, y_train, n_neighbors=borrowed)
    else:
        print_log("Cross-Validation for K-NN...")
        best_knn, _ = run_knn(X_train_processed, y_train, bypass=False)
        if best_knn is not None:
            save_model(best_knn, "knn", use_big_data=USE_BIG_DATA)
        print_log("Best configuration K-NN completed.")

    # --- SVM ---
    print("\n" + "=" * 50)
    print("SVM")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists(svm_model_name, use_big_data=USE_BIG_DATA):
        best_svm = load_model(svm_model_name, use_big_data=USE_BIG_DATA)
    else:
        print_log(f"Cross-Validation for SVM (Fast Mode Linear: {fast_svm_mode})...")
        best_svm, _ = run_svm(
            X_train_processed, y_train, fast_mode=fast_svm_mode
        )
        save_model(best_svm, svm_model_name, use_big_data=USE_BIG_DATA)
        print_log("Best configuration SVM completed.")

    # --- Linear SVM in the balanced arena as well --------------------------
    # Without this the SVM comparison changes kernel and arena at the same
    # time, so the collapse of recall on 'High' in the full-data arena cannot
    # be attributed to either one. Fitting the linear model on the balanced
    # subsample costs a couple of minutes and closes that gap.
    if USE_BIG_DATA:
        best_linear_svm = None
    else:
        print("\n" + "=" * 50)
        print("Linear SVM (balanced arena)")
        print("=" * 50)
        if not FORCE_RETRAIN and model_exists("linear_svm", use_big_data=False):
            best_linear_svm = load_model("linear_svm", use_big_data=False)
        else:
            print_log("Cross-Validation for Linear SVM on the balanced arena...")
            best_linear_svm, _ = run_svm(X_train_processed, y_train, fast_mode=True)
            save_model(best_linear_svm, "linear_svm", use_big_data=False)

    # --- Random Forest ---
    print("\n" + "=" * 50)
    print("Random Forest")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists("random_forest", use_big_data=USE_BIG_DATA):
        best_rf = load_model("random_forest", use_big_data=USE_BIG_DATA)
    else:
        print_log(f"Cross-Validation for Random Forest...")
        best_rf, _ = run_random_forest(X_train_processed, y_train)
        save_model(best_rf, "random_forest", use_big_data=USE_BIG_DATA)
        print_log("Best configuration Random Forest completed.")

    # --- XGBoost ---
    print("\n" + "=" * 50)
    print("XGBoost")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists("xgboost", use_big_data=USE_BIG_DATA):
        best_xg = load_model("xgboost", use_big_data=USE_BIG_DATA)
    else:
        print_log(f"Cross-Validation for XGBoost...")
        best_xg, _ = run_xgboost(X_train_processed, y_train)
        save_model(best_xg, "xgboost", use_big_data=USE_BIG_DATA)
        print_log("Best configuration XGBoost completed.")

    print_log("=== PHASE 4: Final Evaluation on test data ===")
    all_metrics = []
    svm_label = "Linear SVM" if fast_svm_mode else "RBF SVM"
    # Labelled so that no table implies a grid search that never ran.
    knn_label = "K-NN (untuned)" if USE_BIG_DATA else "K-NN"
    models_to_evaluate = [
        (best_majority, "Majority baseline"),
        (best_tree, tree_label),
        (best_knn, knn_label),
        (best_svm, svm_label),
        (best_linear_svm, "Linear SVM"),
        (best_rf, "Random Forest"),
        (best_xg, "XGBoost")
    ]
    predictions = {}

    for model, name in models_to_evaluate:
        if model is not None:
            y_pred, metrics = evaluate_model(
                model, X_test_processed, y_test, model_name=name
            )
            all_metrics.append(metrics)
            predictions[name] = y_pred
            plot_save_confusion_matrix(
                y_test, y_pred, model_name=name, big_data=USE_BIG_DATA
            )

    # Confidence intervals for the comparison itself. The same paired bootstrap
    # the feature ablation uses, with the best-scoring model as the reference,
    # so every delta reads as the distance from the winner. No model is
    # refitted: it resamples the predictions already computed above.
    suffix = "_UsedBigData" if USE_BIG_DATA else ""
    best_name = max(all_metrics, key=lambda m: m["F1-Score"])["Model_name"]
    run_paired_bootstrap(
        y_test,
        predictions,
        reference=best_name,
        filename="model_bootstrap",
        suffix=suffix,
    )
    print_log("=== PHASE 5: Generation of decision boundaries plots (PCA 2D) === ")

    # The tuned estimator itself is passed, not its settings: plot_decision_boundaries_2d
    # clones it, so the plotted model can never disagree with the evaluated one on any
    # hyperparameter.
    if not USE_BIG_DATA:
        plot_decision_boundaries_2d(
            X_train_processed,
            y_train,
            tuned_model=best_knn,
            model_name=f"K-NN (K={best_knn.n_neighbors})",
            model_type="knn",
            big_data=USE_BIG_DATA,
        )
        plot_decision_boundaries_2d(
            X_train_processed,
            y_train,
            tuned_model=best_svm,
            model_name=f"RBF SVM (C={best_svm.C}, gamma={best_svm.gamma})",
            model_type="svm_rbf",
            big_data=USE_BIG_DATA,
        )
    else:
        plot_decision_boundaries_2d(
            X_train_processed,
            y_train,
            tuned_model=best_svm,
            model_name=f"Linear SVM (C={best_svm.C})",
            model_type="linear_svm",
            big_data=USE_BIG_DATA,
        )

    print_log("=== PHASE 5b: Threshold-rule experiment ===")
    run_threshold_rule_experiment(
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        transformer=transformer,
        big_data=USE_BIG_DATA,
        cv_table=depth_cv,
        selected_depth=tree_depth,
    )

    print_log("=== PHASE 5c: Feature-engineering ablation ===")
    run_feature_ablation(
        dataset_path,
        big_data=USE_BIG_DATA,
        sample_per_class=SAMPLE_PER_CLASS,
    )

    print_log("=== PHASE 5d: PCA + K-NN dimensionality sweep ===")
    run_pca_knn_experiment(
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        big_data=USE_BIG_DATA,
    )

    print_log("=== PHASE 5e: Cost-sensitive decision threshold ===")
    run_cost_threshold_experiment(
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        big_data=USE_BIG_DATA,
    )

    print_log("=== PHASE 6: ROC Curve ===")

    feature_names_encoded = transformer.get_feature_names_out()

    models_to_plot = [
        (best_xg, "xgboost"),
        (best_knn, "knn"),
        (best_rf, "rf"),
        (best_svm, "svm"),
        (best_linear_svm, "linear_svm"),
    ]

    auc_rows = []
    for model, name in models_to_plot:
        if model is not None:
            plot_feature_importance(
                model=model,
                feature_names=feature_names_encoded,
                filename=f"feature_importance_{name}.png",
                big_data=USE_BIG_DATA
            )
            auc = plot_multiclass_roc(
                model=model,
                X_test=X_test_processed,
                y_test=y_test,
                filename=f"roc_curve_{name}.png",
                big_data=USE_BIG_DATA
            )
            # The AUCs used to live only in the legend of the PNG, so a figure
            # quoted in the report could not be traced back to a file.
            if auc:
                auc_rows.append({"Model": name, **auc})

    if auc_rows:
        pd.DataFrame(auc_rows).to_csv(
            f"results_notebook/roc_auc{suffix}.csv", index=False, float_format="%.4f"
        )
        print(f"ROC AUCs saved in: results_notebook/roc_auc{suffix}.csv")

    metrics_df = pd.DataFrame(all_metrics)

    # model_comparison.csv keeps its original macro-only schema, so anything
    # already pointing at it does not break; the per-class breakdown, including
    # recall on the minority 'High' class, goes to its own file.
    macro_cols = ["Model_name", "F1-Score", "Accuracy", "Recall", "Precision"]
    metrics_df[macro_cols].to_csv(
        f"results_notebook/model_comparison{suffix}.csv",
        index=False,
        float_format="%.4f",
    )
    metrics_df.to_csv(
        f"results_notebook/per_class_metrics{suffix}.csv",
        index=False,
        float_format="%.4f",
    )

    print_log("=== Pipeline executed with success! ===")


if __name__ == "__main__":
    main()
