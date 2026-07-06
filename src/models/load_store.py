import os
import joblib

MODELS_DIR = os.path.join('saved_models')

def get_model_path(model_name, use_big_data=False):
    """Build the full file path for a saved model."""
    safe_name = model_name.lower().replace(' ', '_')
    dir_name = 'saved_models_BigData' if use_big_data else 'saved_models'
    return os.path.join(dir_name, f"{safe_name}.pkl")

def save_model(model, model_name, use_big_data=False):
    """Save a trained model to disk using joblib."""
    dir_name = 'saved_models_BigData' if use_big_data else 'saved_models'
    os.makedirs(dir_name, exist_ok=True)
    filepath = get_model_path(model_name, use_big_data)
    joblib.dump(model, filepath)
    print(f"  [SAVE] Model saved: {filepath}")

def load_model(model_name, use_big_data=False):
    """
    Load a previously saved model from disk.
    Returns the model if found, None otherwise.
    """
    filepath = get_model_path(model_name, use_big_data)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"  [LOAD] Model loaded from cache: {filepath}")
        return model
    return None

def model_exists(model_name, use_big_data=False):
    """Check if a saved model exists on disk."""
    return os.path.exists(get_model_path(model_name, use_big_data))
