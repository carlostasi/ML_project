from datetime import datetime

def print_log(message):
    """Timestamped log utility for pipeline progress tracking."""
    time = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{time}] {message}")


def get_model_folder_from_name(model_name):
    """Map a human-readable model name (e.g. 'RBF SVM') to its results_notebook subfolder."""
    name = model_name.lower()
    if 'baseline' in name or 'decision tree' in name:
        return 'Baselines'
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
    if 'dummy' in name or 'decisiontree' in name:
        return 'Baselines'
    if 'xgb' in name:
        return 'XGBoost'
    if 'randomforest' in name:
        return 'RandomForest'
    if 'svc' in name:
        return 'SVM'
    if 'neighbors' in name:
        return 'KNN'
    return ''


def slugify_model_name(model_name):
    """
    Turn a display name into a filename-safe slug.

    Display names are allowed to be readable ('Decision Tree (d=8)'), but they
    end up in figure filenames, and those filenames are then passed to LaTeX
    \includegraphics, which cannot handle parentheses or '=' in a path. Keep
    letters, digits, hyphens and underscores; collapse everything else.
    """
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name.lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
