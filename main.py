import os
from src.data.preprocessing import (
    load_data,
    balance_training_set,
    get_pipeline_transformer,
    dataset_setup,
)
from src.models import (
    run_svm,
    run_knn,
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
    run_feature_ablation,
    run_pca_knn_experiment,
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    USE_BIG_DATA = True
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
    best_tree, _ = run_decision_tree(X_train_processed, y_train, max_depth=8)
    save_model(best_tree, "decision_tree", use_big_data=USE_BIG_DATA)

    fast_svm_mode = True if USE_BIG_DATA else False
    svm_model_name = "linear_svm" if fast_svm_mode else "rbf_svm"

    # --- K-NN ---
    print("\n" + "=" * 50)
    print("K-NN")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists("knn", use_big_data=USE_BIG_DATA) and not USE_BIG_DATA:
        best_knn = load_model("knn", use_big_data=USE_BIG_DATA)
        knn_results = None
    elif USE_BIG_DATA:
        print_log("[INFO] K-NN tuning skipped for RAM limits.")
        best_knn, knn_results = None, None
    else:
        print_log("Cross-Validation for K-NN (K=3, 5, )...")
        best_knn, knn_results = run_knn(X_train_processed, y_train, bypass=False)
        if best_knn is not None:
            save_model(best_knn, "knn", use_big_data=USE_BIG_DATA)
        print_log("Best configuration K-NN completed.")

    # --- SVM ---
    print("\n" + "=" * 50)
    print("SVM")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists(svm_model_name, use_big_data=USE_BIG_DATA):
        best_svm = load_model(svm_model_name, use_big_data=USE_BIG_DATA)
        svm_results = None
    else:
        print_log(f"Cross-Validation for SVM (Fast Mode Linear: {fast_svm_mode})...")
        best_svm, svm_results = run_svm(
            X_train_processed, y_train, fast_mode=fast_svm_mode
        )
        save_model(best_svm, svm_model_name, use_big_data=USE_BIG_DATA)
        print_log("Best configuration SVM completed.")

    # --- Random Forest ---
    print("\n" + "=" * 50)
    print("Random Forest")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists("random_forest", use_big_data=USE_BIG_DATA):
        best_rf = load_model("random_forest", use_big_data=USE_BIG_DATA)
        rf_results = None
    else:
        print_log(f"Cross-Validation for Random Forest...")
        best_rf, rf_results = run_random_forest(X_train_processed, y_train)
        save_model(best_rf, "random_forest", use_big_data=USE_BIG_DATA)
        print_log("Best configuration Random Forest completed.")

    # --- XGBoost ---
    print("\n" + "=" * 50)
    print("XGBoost")
    print("=" * 50)
    if not FORCE_RETRAIN and model_exists("xgboost", use_big_data=USE_BIG_DATA):
        best_xg = load_model("xgboost", use_big_data=USE_BIG_DATA)
        xg_results = None
    else:
        print_log(f"Cross-Validation for XGBoost...")
        best_xg, xg_results = run_xgboost(X_train_processed, y_train)
        save_model(best_xg, "xgboost", use_big_data=USE_BIG_DATA)
        print_log("Best configuration XGBoost completed.")

    print_log("=== PHASE 4: Final Evaluation on test data ===")
    all_metrics = []
    svm_label = "Linear SVM" if fast_svm_mode else "RBF SVM"
    models_to_evaluate = [
        (best_majority, "Majority baseline"),
        (best_tree, "Decision Tree (d=8)"),
        (best_knn, "K-NN"),
        (best_svm, svm_label),
        (best_rf, "Random Forest"),
        (best_xg, "XGBoost")
    ]

    for model, name in models_to_evaluate:
        if model is not None:
            y_pred, metrics = evaluate_model(
                model, X_test_processed, y_test, model_name=name
            )
            all_metrics.append(metrics)
            plot_save_confusion_matrix(
                y_test, y_pred, model_name=name, big_data=USE_BIG_DATA
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

    print_log("=== PHASE 6: ROC Curve ===")

    feature_names_encoded = transformer.get_feature_names_out()

    models_to_plot = [
        (best_xg, "xgboost"),
        (best_knn, "knn"),
        (best_rf, "rf"),
        (best_svm, "svm"),
    ]

    for model, name in models_to_plot:
        if model is not None:
            plot_feature_importance(
                model=model,
                feature_names=feature_names_encoded,
                filename=f"feature_importance_{name}.png",
                big_data=USE_BIG_DATA
            )
            plot_multiclass_roc(
                model=model,
                X_test=X_test_processed,
                y_test=y_test,
                filename=f"roc_curve_{name}.png",
                big_data=USE_BIG_DATA
            )
    metrics_df = pd.DataFrame(all_metrics)
    suffix = "_UsedBigData" if USE_BIG_DATA else ""

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
