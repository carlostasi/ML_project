import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from src.utils import get_model_folder_from_name, slugify_model_name
import matplotlib
matplotlib.use('Agg')

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Execute predictions on test set, calculate multiclass metrics and report 
    production.
    """
    print(f"\n--- Performance evaluation for: {model_name} ---")

    # Timed because the models differ by orders of magnitude in what it costs to
    # answer a query, and the report argues about that cost. K-NN in particular
    # does all of its work here: fitting one is free, scoring the test set is not.
    started = time.time()
    y_pred = model.predict(X_test)
    seconds = time.time() - started
    print(f"Prediction over {len(y_test)} rows: {seconds:.1f}s")

    target_names = ['Low', 'Medium', 'High']

    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1-Score Globale: {macro_f1:.4f}")

    accuracy = accuracy_score(y_test, y_pred)
    # zero_division=0 is required by the majority-class baseline, which predicts
    # no samples at all for two of the three classes: without it precision is
    # ill-defined there and sklearn warns instead of scoring it as 0.
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    metrics = {
        "Model_name": model_name,
        "F1-Score": macro_f1,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "Predict_seconds": round(seconds, 1),
    }

    # Per-class figures are kept alongside the macro averages: recall on the
    # minority 'High' class is the operational metric of this project (a missed
    # drought is the costly error), and averaging it away hides the precision
    # collapse that training on a 1:1:1 subsample produces at test time.
    per_class = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    for label in target_names:
        metrics[f"{label}_precision"] = per_class[label]["precision"]
        metrics[f"{label}_recall"] = per_class[label]["recall"]
        metrics[f"{label}_f1"] = per_class[label]["f1-score"]

    return y_pred, metrics
    
def plot_save_confusion_matrix(y_test, y_pred, model_name="Model", output_dir="results_notebook", big_data=False):
    
    target_names = ['Low', 'Medium', 'High']
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)

    # Two separate things used to share one variable here, and the arena label
    # was empty outside big-data mode, which titled every balanced-arena figure
    # "Confusion matrix - K-NN ()". These figures are in the report.
    arena = "entire dataset" if big_data else "balanced subsample"
    plt.title(f"Confusion matrix - {model_name} ({arena})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    folder = get_model_folder_from_name(model_name)
    final_output_dir = os.path.join(output_dir, folder) if folder else output_dir
    os.makedirs(final_output_dir, exist_ok=True)
    suffix = "_UsedBigData" if big_data else ""
    filename = f"confusion_matrix_{slugify_model_name(model_name)}{suffix}.png"
    filepath = os.path.join(final_output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Confusion matrix graph saved in: {filepath}")