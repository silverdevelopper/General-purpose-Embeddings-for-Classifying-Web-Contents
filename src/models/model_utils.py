# src/models/model_utils.py

import joblib

def save_model(model, model_path):
    """
    Save a trained model to a file.

    Args:
        model: The trained model to save.
        model_path (str): Path to save the model file (e.g., .pkl).

    Returns:
        None
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file (e.g., .pkl).

    Returns:
        model: The loaded model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict_model(model, texts):
    """
    Make predictions with a trained model.

    Args:
        model: The trained model to use for predictions.
        texts (list): List of input text data to predict.

    Returns:
        list: List of predictions.
    """
    predictions = model.predict(texts)
    print("Predictions generated successfully.")
    return predictions