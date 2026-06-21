from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def run_knn(X_train, y_train):
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': [3, 5, 7]}

    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.cv_results_

def run_svm(X_train, y_train, fast_mode=True):
    if fast_mode:
        svm = SVC(kernel='linear', cache_size=1000)
        param_grid = {'C': [1.0]}
    else:
        svm = SVC(kernel='rbf', cache_size=1000)
        param_grid = {'C': [1.0], 'gamma': ['scale']}
    
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.cv_results_



