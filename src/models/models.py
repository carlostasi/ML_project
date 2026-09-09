from sklearn.utils import compute_sample_weight
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.models.load_store import load_model, model_exists


def run_knn(X_train, y_train, bypass=False, n_jobs=-1):
    """
    Grid search over K.

    `n_jobs` is the parallelism of the *search*, not of the distance computation.
    On the full partition it should be 1: a single pass already uses every core
    through scikit-learn's OpenMP distance kernel, so running candidates in
    parallel buys no speed at all and only multiplies the working memory. Thirty
    passes over 504,000 training rows take about a quarter of an hour either way.
    """
    if bypass:
        print("[INFO] K-NN tuning skipped for RAM limits.")
        return None, None
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 31, 51, 101, 151, 200]}

    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='f1_macro', n_jobs=n_jobs, verbose=3)
    grid_search.fit(X_train, y_train)

    # Print best model parameters
    print(f"\n >>> BEST configuration K-NN: {grid_search.best_params_}")
    print(f" >>> Average score in Cross-Validation (Macro F1): {grid_search.best_score_:.4f}\n")
    return grid_search.best_estimator_, grid_search.cv_results_

def run_svm(X_train, y_train, fast_mode=True):
    if fast_mode:
        svm = LinearSVC(C=1.0, dual=False, random_state=42, class_weight='balanced', max_iter=2000, verbose=True)
        param_grid = {'C': [0.1, 1.0, 10.0]}

        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=3)
        grid_search.fit(X_train, y_train)

        # Print best model parameters
        print(f"\n >>> BEST configuration LINEAR SVM: {grid_search.best_params_}")
        print(f" >>> Average score in Cross-Validation (Macro F1): {grid_search.best_score_:.4f}\n")
        return grid_search.best_estimator_, grid_search.cv_results_
    else:
        svm = SVC(kernel='rbf', cache_size=2000)
        param_grid = {'C': [0.1, 1.0, 10.0], 
                      'gamma': ['scale', 'auto']}
    
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    # Print best model parameters
    print(f"\n >>> BEST configuration RBF SVM: {grid_search.best_params_}")
    print(f" >>> Average score in Cross-Validation (Macro F1): {grid_search.best_score_:.4f}\n")
    return grid_search.best_estimator_, grid_search.cv_results_

def run_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    
    print(f"\n >>> BEST configuration RANDOM FOREST: {grid_search.best_params_}")
    print(f" >>> Average score in Cross-Validation (Macro F1): {grid_search.best_score_:.4f}\n")
    return grid_search.best_estimator_, grid_search.cv_results_

def run_xgboost(X_train, y_train):
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    param_grid = {
        'n_estimators': [100, 150, 200],  # Numero di alberi sequenziali da costruire
        'max_depth': [4, 6],              # Profondità massima dell'albero (più è alto, più rischia overfitting)
        'learning_rate': [0.05, 0.1],     # Passo di sintonizzazione del gradiente (shrunk factor)
        'subsample': [0.8, 1.0]           # Percentuale di campionamento delle righe per ogni albero
    }
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=3
    )
    
    # 5. Esecuzione del fit iniettando i pesi calcolati
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"\n >>> BEST configuration XGBoost: {grid_search.best_params_}")
    print(f" >>> Average score in Cross-Validation (Macro F1): {grid_search.best_score_:.4f}\n")
    return grid_search.best_estimator_, grid_search.cv_results_




# --- Baselines -------------------------------------------------------------
# Both are deliberately left un-tuned: their job is to put the headline score of
# the four tuned models in context, so a GridSearchCV here would defeat the
# purpose. They return (estimator, None) to match the (best_estimator_,
# cv_results_) signature of the run_<model> functions above.

def tuned_k(use_big_data=False, default=31):
    """
    Read K back from a tuned K-NN, so that no caller has to hardcode it.

    Falls back to the balanced arena when the requested one has no pickle, which
    is what the full-data arena needed before it had a search of its own. Used by
    the PCA sweep, which must run at the K of the arena it is sweeping.
    """
    if model_exists("knn", use_big_data=use_big_data):
        return load_model("knn", use_big_data=use_big_data).get_params()["n_neighbors"]
    if use_big_data and model_exists("knn", use_big_data=False):
        return load_model("knn", use_big_data=False).get_params()["n_neighbors"]
    return default


def run_majority_baseline(X_train, y_train):
    """
    Lower bound: always predict the majority class ('Low', 58.72% of the test
    set). It never detects a drought at all, so it fixes the floor against
    which recall on 'High' has to be read.
    """
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X_train, y_train)
    print("\n >>> Majority-class baseline fitted (no tuning by design).\n")
    return model, None


def run_decision_tree(X_train, y_train, max_depth=8, class_weight=None):
    """
    Interpretable reference: a single axis-aligned tree, and at the right depth
    the match for a 300-tree Random Forest on macro F1 while being four orders of
    magnitude smaller — the central result of the project.

    max_depth is expected to come from select_tree_depth in
    src/experiments/threshold_rule.py, which picks it by cross-validation on the
    training partition; main.py passes it explicitly. The default of 8 is only a
    fallback for a direct call. Earlier versions hardcoded 8 because that is
    where the depth sweep peaks on the full-data arena, but that sweep is scored
    on the test set, so using it to choose the depth would have selected a model
    on the data the same model is then reported against. The selected depth may
    now differ between arenas, which is expected: the recoverable depth grows
    with the training sample.

    class_weight is left at None on purpose, unlike Random Forest and XGBoost.
    The baseline is meant to be the plainest possible tree, and weighting it
    would make it a design choice rather than a reference point. The weighting
    is not free either: measured on the full-data arena it trades macro F1 for
    recall on the minority class (0.9682 -> 0.9606 macro F1, 0.9074 -> 0.9262
    recall on 'High'). In the balanced arena it changes nothing, the training
    subsample already being 1:1:1.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth, class_weight=class_weight, random_state=42
    )
    model.fit(X_train, y_train)
    print(f"\n >>> Decision Tree baseline fitted (max_depth={max_depth}, "
          f"{model.get_n_leaves()} leaves, no tuning by design).\n")
    return model, None
