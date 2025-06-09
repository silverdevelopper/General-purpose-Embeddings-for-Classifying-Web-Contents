# Evaluations Folder

## Purpose
The `evaluations/` folder is responsible for evaluating and analyzing the performance of models across experiments. It includes scripts for:
- Calculating key metrics (e.g., accuracy, precision, recall, F1-score).
- Generating confusion matrices.
- Aggregating results from multiple experiments.
- Visualizing and comparing experiment performance.

---

### Folder Overview

1. `metrics.py`: Calculate evaluation metrics like accuracy, precision, recall, and F1-score.
2. `confusion_matrix.py`: Generate and save confusion matrix visualizations.
3. `results_aggregator.py`: Aggregate evaluation results from multiple experiment runs.
4. `visualize_results.py`: Generate comparison plots based on aggregated results.

---

### Example Usage

1. **Evaluate Model Metrics**
   ```bash
   python evaluations/metrics.py --true_path data/splits/test_labels.csv --pred_path results/predictions.csv --output_path results/evaluation.json