from sklearn.utils import compute_sample_weight
from prompt_toolkit.key_binding.bindings.scroll import scroll_forward
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def run_knn(X_train, y_train, bypass=False):
    if bypass:
        print("[INFO] K-NN tuning skipped for RAM limits.")
        return None, None
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 31, 51, 101, 151, 200]}

    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=3)
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
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300]}
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    
    print(f"\n >>> BEST configuration RANDOM FOREST: {grid_search.best_params_}")
    print(f" >>> Average score in Cross-Validation (Macro F1): {grid_search.best_score_:.4f}\n")
    return grid_search.best_estimator_, grid_search.cv_results_

def run_xgboost(X_train, y_train):
    xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss')
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


