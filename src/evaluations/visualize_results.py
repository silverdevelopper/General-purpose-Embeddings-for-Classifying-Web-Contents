# evaluations/visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_comparison(aggregated_file, metric, save_path):
    """
    Create a bar plot to compare a specific evaluation metric across experiments.

    Args:
        aggregated_file (str): Path to the aggregated results CSV file.
        metric (str): The metric to compare (e.g., accuracy, f1_score).
        save_path (str): Path to save the comparison plot.
    """
    df = pd.read_csv(aggregated_file)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="experiment", y=metric, data=df, palette="muted")
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Comparison of {metric.capitalize()} Across Experiments")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Experiment")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize experiment results.")
    parser.add_argument("--aggregated_file", type=str, required=True, help="Path to aggregated results CSV file.")
    parser.add_argument("--metric", type=str, required=True, help="Metric to visualize (e.g., accuracy, f1_score).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the visualization plot.")

    args = parser.parse_args()
    plot_metric_comparison(args.aggregated_file, args.metric, args.output_path)