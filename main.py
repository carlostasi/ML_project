import os
from datetime import datetime
from src.data.preprocessing import load_data, get_pipeline_transformer, dataset_setup
from src.models.models import run_svm, run_knn, run_xgboost, run_random_forest
from src.models.load_store import save_model, load_model, model_exists
from src.metrics.evaluation import evaluate_model, plot_save_confusion_matrix
from src.metrics.visualization import plot_decision_boundaries_2d, run_knn_with_pca_experiment, plot_correlation_matrix, plot_feature_importance, plot_multiclass_roc

def print_log(message):
    time = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{time}] {message}")

def main():
    USE_BIG_DATA = True
    FORCE_RETRAIN = False  # Set to True to ignore saved models and retrain from scratch

    dataset_path = os.path.join('data', 'train.csv')

    if not os.path.exists(dataset_path):
        print_log(f"Error: Dataset file not found")
        return

    print_log("=== PHASE 1: Loading data & Feature Engineering??? ===")
    sample_size = None if USE_BIG_DATA else 21000
    X, y = load_data(dataset_path, sample_size_per_class=sample_size)
    print_log(f"Dataset ready. Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    # print("Colonne effettive in X:", X.columns.tolist())

    print_log("=== PHASE 2: Split and Preprocessing ===")
    X_train, X_test, y_train, y_test = dataset_setup(X, y)
    plot_correlation_matrix(X_train)

    X_train = X_train.drop(columns=['Rainfall_mm', 'Previous_Irrigation_mm'])
    X_test = X_test.drop(columns=['Rainfall_mm', 'Previous_Irrigation_mm'])

    transformer = get_pipeline_transformer()
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)
    print_log(f"Pre-processing completed. Feature post-encoding: {X_train_processed.shape[1]}")

    print_log("=== PHASE 3: Training and tuning models ===")
    fast_svm_mode = True if USE_BIG_DATA else False
    svm_model_name = "linear_svm" if fast_svm_mode else "rbf_svm"

    # --- K-NN ---
    print("\n" + "="*50)
    print("K-NN")
    print("="*50)
    if not FORCE_RETRAIN and model_exists("knn") and not USE_BIG_DATA:
        best_knn = load_model("knn")
        knn_results = None
    elif USE_BIG_DATA:
        print_log("[INFO] K-NN tuning skipped for RAM limits.")
        best_knn, knn_results = None, None
    else:
        print_log("Cross-Validation for K-NN (K=3, 5, )...")
        best_knn, knn_results = run_knn(X_train_processed, y_train, bypass=False)
        if best_knn is not None:
            save_model(best_knn, "knn")
        print_log("Best configuration K-NN completed.")

    # --- SVM ---
    print("\n" + "="*50)
    print("SVM")
    print("="*50)
    if not FORCE_RETRAIN and model_exists(svm_model_name):
        best_svm = load_model(svm_model_name)
        svm_results = None
    else:
        print_log(f"Cross-Validation for SVM (Fast Mode Linear: {fast_svm_mode})...")
        best_svm, svm_results = run_svm(X_train_processed, y_train, fast_mode=fast_svm_mode)
        save_model(best_svm, svm_model_name)
        print_log("Best configuration SVM completed.")

    # --- Random Forest ---
    print("\n" + "="*50)
    print("Random Forest")
    print("="*50)
    if not FORCE_RETRAIN and model_exists("random_forest"):
        best_rf = load_model("random_forest")
        rf_results = None
    else:
        print_log(f"Cross-Validation for Random Forest...")
        best_rf, rf_results = run_random_forest(X_train_processed, y_train)
        save_model(best_rf, "random_forest")
        print_log("Best configuration Random Forest completed.")

    # --- XGBoost ---
    print("\n" + "="*50)
    print("XGBoost")
    print("="*50)
    if not FORCE_RETRAIN and model_exists("xgboost"):
        best_xg = load_model("xgboost")
        xg_results = None
    else:
        print_log(f"Cross-Validation for XGBoost...")
        best_xg, xg_results = run_xgboost(X_train_processed, y_train)
        save_model(best_xg, "xgboost")
        print_log("Best configuration XGBoost completed.")

    print_log("=== PHASE 4: Final Evaluation on test data ===")
    # KNN
    if best_knn is not None:
        y_pred_knn = evaluate_model(best_knn, X_test_processed, y_test, model_name="K-NN")
        plot_save_confusion_matrix(y_test, y_pred_knn, model_name="K-NN", big_data=USE_BIG_DATA)

    # SVM
    svm_label = "Linear SVM" if fast_svm_mode else "RBF SVM"
    y_pred_svm = evaluate_model(best_svm, X_test_processed, y_test, model_name=svm_label)
    plot_save_confusion_matrix(y_test, y_pred_svm, model_name=svm_label, big_data=USE_BIG_DATA)

    # Random Forest
    y_pred_rf = evaluate_model(best_rf, X_test_processed, y_test, model_name="Random Forest")
    plot_save_confusion_matrix(y_test, y_pred_rf, model_name="Random Forest", big_data=USE_BIG_DATA)

    # XGBoost
    y_pred_xg = evaluate_model(best_xg, X_test_processed, y_test, model_name="XGBoost")
    plot_save_confusion_matrix(y_test, y_pred_xg, model_name="XGBoost", big_data=USE_BIG_DATA)

    print_log("=== PHASE 5: Generation of decision boundaries plots (PCA 2D) === ")

    if not USE_BIG_DATA:
        plot_decision_boundaries_2d(X_train_processed, y_train, model_name="K-NN (K=31)", model_type="knn", knn_k=31)
        plot_decision_boundaries_2d(X_train_processed, y_train, model_name="RBF SVM (C=10)", model_type="svm_rbf", svm_c=10.0)
    else:
        plot_decision_boundaries_2d(X_train_processed, y_train, model_name="LINEAR SVM", model_type="linear_svm", svm_c=1.0)

    print("\n=== Run experiment (PCA + K-NN) ===")
    # Lanciamo il test riducendo lo spazio a 3 componenti principali per vedere l'effetto sul K-NN
    if not USE_BIG_DATA:
        run_knn_with_pca_experiment(X_train_processed, y_train, X_test_processed, y_test, n_neighbors=31, n_components=3)

    print_log("=== PHASE 6: ROC Curve ===")

    feature_names_encoded = transformer.get_feature_names_out()

    plot_feature_importance(
        model=best_xg,
        feature_names=feature_names_encoded,
        filename="feature_importance_xgboost.png"
    )

    plot_multiclass_roc(
        model=best_xg,
        X_test=X_test_processed,
        y_test=y_test,
        filename="roc_curve_xgboost.png"
    )


    print_log("=== Pipeline executed with success! ===")

if __name__ == '__main__':
    main()