import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

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

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nRecall: {recall:.4f}")
    print(f"\nPrecision: {precision:.4f}")
    
    return y_pred 
    
def plot_save_confusion_matrix(y_test, y_pred, model_name="Model", output_dir="results_notebook", big_data=False):
    
    target_names = ['Low', 'Medium', 'High']
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)

    plt.title(f"Confusion matrix - {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    if big_data:
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_UsedBigData.png"
    else:
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Confusion matrix graph saved in: {filepath}")