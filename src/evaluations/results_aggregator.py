# evaluations/results_aggregator.py

import os
import pandas as pd
import json

def aggregate_results(results_dir, output_file):
    """
    Aggregate evaluation metrics from multiple JSON files into a single CSV for comparison.

    Args:
        results_dir (str): Directory containing JSON result files.
        output_file (str): Path to save the aggregated results as a CSV file.
    """
    aggregated_data = []
    
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(results_dir, file_name), "r") as f:
                metrics = json.load(f)
                metrics["experiment"] = file_name.replace(".json", "")
                aggregated_data.append(metrics)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(aggregated_data)
    df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics across experiments.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing JSON evaluation results.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save aggregated results as a CSV.")
    
    args = parser.parse_args()
    aggregate_results(args.results_dir, args.output_file)