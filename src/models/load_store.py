import os
import joblib

MODELS_DIR = os.path.join('saved_models')

def get_model_path(model_name):
    """Build the full file path for a saved model."""
    safe_name = model_name.lower().replace(' ', '_')
    return os.path.join(MODELS_DIR, f"{safe_name}.pkl")

def save_model(model, model_name):
    """Save a trained model to disk using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = get_model_path(model_name)
    joblib.dump(model, filepath)
    print(f"  [SAVE] Model saved: {filepath}")

def load_model(model_name):
    """
    Load a previously saved model from disk.
    Returns the model if found, None otherwise.
    """
    filepath = get_model_path(model_name)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"  [LOAD] Model loaded from cache: {filepath}")
        return model
    return None

def model_exists(model_name):
    """Check if a saved model exists on disk."""
    return os.path.exists(get_model_path(model_name))
