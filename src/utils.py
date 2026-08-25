from datetime import datetime

def print_log(message):
    """Timestamped log utility for pipeline progress tracking."""
    time = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{time}] {message}")


def get_model_folder_from_name(model_name):
    """Map a human-readable model name (e.g. 'RBF SVM') to its results_notebook subfolder."""
    name = model_name.lower()
    if 'knn' in name or 'k-nn' in name:
        return 'KNN'
    if 'svm' in name:
        return 'SVM'
    if 'random forest' in name or 'rf' in name:
        return 'RandomForest'
    if 'xgboost' in name:
        return 'XGBoost'
    return ''


def get_model_folder_from_model(model):
    """Map a fitted estimator instance to its results_notebook subfolder."""
    name = type(model).__name__.lower()
    if 'xgb' in name:
        return 'XGBoost'
    if 'randomforest' in name:
        return 'RandomForest'
    if 'svc' in name:
        return 'SVM'
    if 'neighbors' in name:
        return 'KNN'
    return ''
