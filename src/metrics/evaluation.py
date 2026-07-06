import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import matplotlib
matplotlib.use('Agg')

def get_model_folder(model_name):
    name = model_name.lower()
    if 'knn' in name or 'k-nn' in name:
        return 'KNN'
    elif 'svm' in name:
        return 'SVM'
    elif 'random forest' in name or 'rf' in name:
        return 'RandomForest'
    elif 'xgboost' in name:
        return 'XGBoost'
    return ''

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Execute predictions on test set, calculate multiclass metrics and report 
    production.
    """
    print(f"\n--- Performance evaluation for: {model_name} ---")

    y_pred = model.predict(X_test)

    target_names = ['Low', 'Medium', 'High']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1-Score Globale: {macro_f1:.4f}")

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    return y_pred, {"Model_name": model_name, "F1-Score": macro_f1, "Accuracy": accuracy, "Recall": recall, "Precision": precision} 
    
def plot_save_confusion_matrix(y_test, y_pred, model_name="Model", output_dir="results_notebook", big_data=False):
    
    target_names = ['Low', 'Medium', 'High']
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)

    suffix = "- Entire dataset" if big_data else ""
    plt.title(f"Confusion matrix - {model_name} ({suffix})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    folder = get_model_folder(model_name)
    final_output_dir = os.path.join(output_dir, folder) if folder else output_dir
    os.makedirs(final_output_dir, exist_ok=True)
    if big_data:
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_UsedBigData.png"
    else:
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(final_output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Confusion matrix graph saved in: {filepath}")