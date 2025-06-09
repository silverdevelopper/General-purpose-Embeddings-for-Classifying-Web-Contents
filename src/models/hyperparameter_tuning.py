# src/models/hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd

def tune_hyperparameters(X_train, y_train, X_test, y_test, param_grid, scoring="accuracy", cv=3, verbose=1):
    """
    Perform hyperparameter tuning on an XGBoost model using GridSearchCV.

    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training labels.
        X_test (array-like): Testing feature matrix.
        y_test (array-like): Testing labels.
        param_grid (dict): Hyperparameter grid for tuning.
        scoring (str): Scoring metric for evaluation (default: "accuracy").
        cv (int): Number of cross-validation folds (default: 3).
        verbose (int): Verbosity level.

    Returns:
        dict: Best parameters from GridSearchCV.
    """
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    print("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, verbose=verbose)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print("Evaluating the tuned model on the test set...")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy with tuned parameters: {test_accuracy}")

    return grid_search.best_params_, test_accuracy

if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Perform hyperparameter tuning for XGBoost.")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to feature matrix (CSV file).")
    parser.add_argument("--label_path", type=str, required=True, help="Path to label file (CSV file).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the best hyperparameters (JSON file).")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X = pd.read_csv(args.embedding_path).values
    y = pd.read_csv(args.label_path)["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
    }

    # Perform hyperparameter tuning
    best_params, test_accuracy = tune_hyperparameters(X_train, y_train, X_test, y_test, param_grid)

    # Save results
    import json
    results = {"best_params": best_params, "test_accuracy": test_accuracy}
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Best parameters and accuracy saved to {args.output_path}")