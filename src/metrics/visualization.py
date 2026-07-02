from sklearn.metrics import accuracy_score, recall_score, precision_score
from multiprocessing.sharedctypes import Value
from sklearn.svm import SVC, LinearSVC
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_decision_boundaries_2d(X_train, y_train, model_name, model_type, knn_k=31, svm_c=10.0, output_dir="plots_notebook"):
    from main import print_log
    print_log(f"Loading decision boundary plot for: {model_name}")
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)

    if model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=knn_k)
    elif model_type == "svm_rbf":
        model = SVC(kernel='rbf', C=svm_c, gamma='scale', random_state=42)
    elif model_type == "linear_svm":
        model = LinearSVC(C=svm_c, dual=False, class_weight='balanced', random_state=42)
    else:
        raise ValueError("Model not supported for plot generation.")

    model.fit(X_train_pca, y_train)

    x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
    y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5

    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#A0C4FF', '#CAFFBF', '#FFADAD']) 
    cmap_bold = ['#001CF0', '#00B000', '#FF0000']                
    
    plt.figure(figsize=(9, 7))
    
    # Disegna le zone colorate di background
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    
    # Sovrappone i punti reali del dataset (ne stampiamo un subset per non affollare il grafico)
    subset_size = min(2000, len(X_train_pca))
    indices = np.random.choice(len(X_train_pca), subset_size, replace=False)
    
    target_names = ['Low', 'Medium', 'High']
    for class_idx, color in enumerate(cmap_bold):
        class_mask = (y_train.iloc[indices] == class_idx) if hasattr(y_train, 'iloc') else (y_train[indices] == class_idx)
        plt.scatter(X_train_pca[indices][class_mask, 0], 
                    X_train_pca[indices][class_mask, 1], 
                    c=color, label=target_names[class_idx],
                    edgecolor='k', s=25, alpha=0.7)
    
    plt.title(f'Decision boundary 2D (PCA 2D) - {model_name}')
    plt.xlabel('Principal Component 1 (PC1)')
    plt.ylabel('Principal component 2 (PC2)')
    plt.legend(loc='upper right', title="Real classes")
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"decision_boundary_{model_type}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Plot saved with success in: {filepath}")

def run_knn_with_pca_experiment(X_train, y_train, X_test, y_test, n_neighbors=31, n_components=3):
    
    print(f"\n=== Esperiment: PCA (Dim={n_components}) + K-NN (K={n_neighbors}) ===")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f" -> Total Variance explained by the {n_components} components: {variance_explained:.2f}%")
    
    knn_pca = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    knn_pca.fit(X_train_pca, y_train)
    y_pred = knn_pca.predict(X_test_pca)
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n--- Performance Report (PCA + K-NN) ---")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    print(f"Global Macro F1-Score: {macro_f1:.4f}")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    
    
    return macro_f1

def plot_correlation_matrix(df, output_dir="results_notebook"):
    print(" Loading correlation matrix...")

    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 8}
    )

    plt.title("Correlation Matrix - Feature Space", fontsize=14, pad=20)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Correlation Matrix saved with success in: {filepath}")

def plot_feature_importance(model, feature_names, outout_dir="results_notebook", filename="feature_importance.png"):
    if not hasattr(model, "feature_importances_"):
        print(f"[WARNING] Model {type(model).__name__} doesn't support the feature importance. Plot skipped.")
        return
    
    print(f"Generation Feature Importance for {type(model).__name__}...")
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], align="center", color="teal")
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title(f"Feature Importance Matrix - {type(model).__name__}", fontsize=12, pad=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(outout_dir, exist_ok=True)
    filepath = os.path.join(outout_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Feature Importance saved: {filepath}")

def plot_multiclass_roc(model, X_test, y_test, classes=['Low', 'Medium', 'High'], output_dir="results_notebook", filename="roc_curve.png"):
    print(f"Generation ROC curve for {type(model).__name__}...")

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"[WARNING] Model {type(model).__name__} doesn't support the calculation of the probabilities. ROC skipped.")
        return

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['royalblue', 'forestgreen', 'crimson']

    for i, color in enumerate(colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f"ROC curve: {classes[i]} (AUC = {roc_auc[i]:.4f})"
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title(f"Receiver Operating Characteristic (OvR) - {type(model).__name__}", fontsize=12, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"ROC Cruve saved: {filepath}")