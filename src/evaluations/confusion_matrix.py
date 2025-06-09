# evaluations/confusion_matrix.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Generate and save a confusion matrix as an image.

    Args:
        y_true (list/array): Ground truth labels.
        y_pred (list/array): Predicted labels by the model.
        labels (list): List of class labels for the confusion matrix.
        save_path (str): File path to save the confusion matrix image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and save a confusion matrix.")
    parser.add_argument("--true_path", type=str, required=True, help="Path to ground truth labels (CSV file).")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted labels (CSV file).")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="List of class labels.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the confusion matrix image.")
    
    args = parser.parse_args()

    # Load ground truth and predictions
    y_true = pd.read_csv(args.true_path)["label"]  # Adjust column name if needed
    y_pred = pd.read_csv(args.pred_path)["prediction"]  # Adjust column name if needed

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_true, y_pred, args.labels, args.output_path)