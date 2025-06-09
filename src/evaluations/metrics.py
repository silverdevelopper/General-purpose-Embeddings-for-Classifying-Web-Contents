# evaluations/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classifier(y_true, y_pred):
    """
    Calculate and return key evaluation metrics for classification.

    Args:
        y_true (list/array): Ground truth labels.
        y_pred (list/array): Predicted labels by the model.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    return metrics

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Evaluate a classifier's predictions.")
    parser.add_argument("--true_path", type=str, required=True, help="Path to ground truth labels (CSV file).")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted labels (CSV file).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save evaluation metrics (JSON file).")
    
    args = parser.parse_args()
    
    # Load ground truth and predictions
    y_true = pd.read_csv(args.true_path)["label"]  # Adjust column name if needed
    y_pred = pd.read_csv(args.pred_path)["prediction"]  # Adjust column name if needed

    # Calculate evaluation metrics
    metrics = evaluate_classifier(y_true, y_pred)
    
    # Save metrics to a JSON file
    import json
    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation metrics saved to {args.output_path}")