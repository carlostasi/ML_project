import os
from datetime import datetime
from src.data.preprocessing import load_data, get_pipeline_transformer, dataset_setup
from src.models.models import run_svm, run_knn
from src.metrics.evaluation import evaluate_model, plot_save_confusion_matrix

def print_log(message):
    time = datetime.now().strftime("%H:%M:%S")
    print(f"[{time}] {message}")

def main():

    USE_BIG_DATA = True

    dataset_path = os.path.join('data', 'train.csv')

    if not os.path.exists(dataset_path):
        print_log(f"Error: Dataset file not found")
        return

    print_log("=== PHASE 1: Loading data & Feature Engineering??? ===")
    sample_size = None if USE_BIG_DATA else 35000
    X, y = load_data(dataset_path, sample_size_per_class=sample_size)
    print_log(f"Dataset ready. Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    # print(f"Dataset ready. Rows: {X.shape[0]}, Columns: {X.shape[1]}")

    print_log("\n=== PHASE 2: Split and Preprocessing ===")
    X_train, X_test, y_train, y_test = dataset_setup(X, y)

    transformer = get_pipeline_transformer()
    X_train_processed = transformer.fit_transform(X_train)
    X_test_processed = transformer.transform(X_test)
    print_log(f"Pre-processing completed. Feature post-encoding: {X_train_processed.shape[1]}")

    print_log("\n=== PHASE 3: Training and tuning K-NN ===")
    print_log("Cross-Validation per K-NN (K=3, 5, )...")
    best_knn, knn_results = run_knn(X_train_processed, y_train)
    print_log("Best configuration K-NN completed.")

    print_log("\n=== PHASE 4: Training and tuning SVM ===")
    fast_svm_mode = True if USE_BIG_DATA else False
    print_log(f"Cross-Validation for SVM (Fast Mode Linear: {fast_svm_mode})...")
    best_svm, svm_results = run_svm(X_train_processed, y_train, fast_mode=fast_svm_mode)
    print_log("Best configuration SVM completed.")

    print_log("\n=== PHASE 5: Final Evaluation on test data ===")
    # KNN
    y_pred_knn = evaluate_model(best_knn, X_test_processed, y_test, model_name="K-NN")
    plot_save_confusion_matrix(y_test, y_pred_knn, model_name="K-NN")

    # SVM
    svm_label = "Linear SVM" if fast_svm_mode else "RBF SVM"
    y_pred_svm = evaluate_model(best_svm, X_test_processed, y_test, model_name=svm_label)
    plot_save_confusion_matrix(y_test, y_pred_svm, model_name=svm_label)

    print_log("\n=== Pipeline executed with success! ===")

if __name__ == '__main__':
    main()