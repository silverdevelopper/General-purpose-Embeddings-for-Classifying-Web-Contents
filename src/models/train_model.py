import sys
sys.path.append("./src/")
import json
import os
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import time
import matplotlib.pyplot as plt
from embeddings.base_embedding import load_embeddings as loader
import argparse
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from model_utils import save_model
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def save_classification_report(y_val, y_pred, accuracy, classifier, params, classification_report_path, grid_search=None,roc_auc_score=None):
    report_dict = classification_report(y_val, y_pred, output_dict=True)
    report_dict["accuracy"] = accuracy
    report_dict["roc_auc"] = roc_auc_score
    report_dict["model"] = classifier
    report_dict["params"] = grid_search.best_params_ if grid_search else params
    report_dict["mode"] = "training"

    # Save the classification report to a JSON file
    with open(classification_report_path, "w") as f:
        # Save with indentation for readability
        json.dump(report_dict, f, indent=4)

    print(f"Classification report saved to {classification_report_path}")

def plot_multiclass_roc(y_true, y_prob, n_classes, dataset_label='Validation',path:str=None):
    """
    Plots the ROC curve for a multiclass classification task and calculates the AUC for each class.

    Parameters:
    - y_true: array-like, shape (n_samples,)
        True labels for each sample.
    - y_prob: array-like, shape (n_samples, n_classes)
        Target scores, which can be probability estimates of the positive class for each sample.
    - n_classes: int
        Number of classes in the target.
    - dataset_label: str, optional (default='Validation')
        Label for the dataset being plotted (e.g., 'Validation', 'Test').
    """
    # Binarize the true labels for multiclass ROC
    if n_classes == 2:
        # For binary classification, create manual binarization
        y_true_bin = np.zeros((len(y_true), 2))
        for i, label in enumerate(y_true):
            y_true_bin[i, int(label)] = 1
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'ROC curve for class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve - {dataset_label}')
    plt.legend(loc='lower right')
    plt.grid()
    if path:
        path=path.replace("learning_curve","roc_curve")
        print(f"ROC curve saved to {path}")
        plt.savefig(path)
    else:
        plt.show()
    return roc_auc

def visualize_model(xgb_model:xgb.XGBClassifier, chart_path=None, classifier="XGBClassifier"):
    results = xgb_model.evals_result()  # Get the evaluation results during training

    print("Keys in evals_result_:", results.keys())
    # Epoch-wise loss on validation set
    validation_loss = results['validation_0']['mlogloss']
    epochs = range(1, len(validation_loss) + 1)  # Epoch numbers
    # Plot the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
    plt.title('Validation Loss Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid()

    if chart_path:
        plt.savefig(chart_path)
        print(f"Learning curve saved to {chart_path}")


def load_split_labels(split_file, label_encoder=None, label_column="label"):
    """
    Load labels for a given train or test split file.

    Args:
        split_file (str): Path to the train/test split file.
        label_encoder (LabelEncoder): The encoder for converting categorical labels to numeric.

    Returns:
        tuple: Encoded labels, indices of the split file.
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    print(f"Loading split labels from {split_file}...")
    split_data = pd.read_csv(split_file)
    labels = split_data[label_column]
    if label_encoder is not None:
        # Encode categorical labels into integers
        labels = label_encoder.transform(labels)
    return labels, split_data.index

# === Embedding Loaders ===


def load_embeddings(path=None, embedding_type="tfidf", dataset_name="ds.csv", embeddings_dir="data/embeddings", label_column="label",store_type="csv"):
    """
    Dynamically load embeddings based on embedding type and dataset.

    Args:
        path (str): Path to the embedding file. If provided, this will be used instead of the constructed path.
        embedding_type (str): Type of embedding (e.g., doc2vec, tfidf, fasttext, huggingface, tfidf_cf).
        dataset_name (str): Dataset name used to locate embedding files.
        embeddings_dir (str): Path to the directory where embeddings are stored.

    Returns:
        numpy.ndarray: Loaded embedding matrix.
    """
    if path:
        embedding_file = path
    else:
        embedding_file = f"{embeddings_dir}/{embedding_type}_{dataset_name}_embeddings.{store_type}"
    if not os.path.exists(embedding_file) and not os.path.exists(embedding_file.replace(".csv", ".npz")):
        raise FileNotFoundError(
            f"Embedding file not found at {embedding_file}. Make sure to run the generate_embeddings.sh script first.")
    print(f"Loading embeddings from {embedding_file}...")
    return loader(embedding_file, label_column=label_column,store_type=store_type)


# === Model Training and Evaluation ===

def train_and_evaluate(X_train, y_train, X_val=None, y_val=None, params={}, save_path=None, 
                      classification_report_path=None, chart_path=None, classifier="XGBClassifier",
                      labels:list=[], use_gpu:bool=True):
    """
    Train a classifier and evaluate on the test set with optional GPU support.

    Args:
        X_train (array-like): Training dataset features.
        y_train (array-like): Training dataset labels.
        X_val (array-like): Validation dataset features.
        y_val (array-like): Validation dataset labels.
        params (dict): Classifier hyperparameters.
        save_path (str): File path to save the trained model.
        classification_report_path (str): Path to save classification report.
        chart_path (str): Path to save charts.
        classifier (str): Type of classifier to use.
        labels (list): List of class labels.
        use_gpu (bool): Whether to use GPU acceleration if available.

    Returns:
        None
    """
    print("Hyperparameters:", params)
    grid_search = params.get("grid_search", False)
    roc_auc_score = None
    normalize = params.get("normalize", False)
    
    # Check GPU availability
    if use_gpu:
        try:
            import cupy as cp
            from cuml.svm import SVC as cuSVC
            from cuml.linear_model import LogisticRegression as cuLogisticRegression
            from cuml.preprocessing import StandardScaler as cuStandardScaler
            GPU_AVAILABLE = True
            print("GPU acceleration enabled")
        except ImportError:
            GPU_AVAILABLE = False
            print("GPU libraries not found. Falling back to CPU.")
    else:
        GPU_AVAILABLE = False

    # Define common param grids for grid search
    param_grids = {
        "LogisticRegression": {
            'max_iter': [100, 1000, 4000],
            'C': [0.5, 0.7, 0.9, 1],
            'tol': [1e-3, 1e-4, 1e-5],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        "SVM": {
            'C': [0.5, 0.7, 0.9, 1],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'tol': [1e-2, 1e-3, 1e-4],
        },
        "XGBClassifier": {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 300, 600],
            'gamma': [0, 0.1, 0.2]
        }
    }

    # Initialize model based on classifier type
    if classifier == "XGBClassifier":
        # XGBoost GPU configuration
        if GPU_AVAILABLE and use_gpu:
            params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0
            })
        
        base_model = xgb.XGBClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            n_estimators=params.get("n_estimators", 100),
            objective=params.get("objective", "multi:softmax"),
            use_label_encoder=False,
            num_class=params.get("num_class", len(labels)),
            eval_metric=params.get("eval_metric", "mlogloss"),
            gamma=params.get("gamma", 0),
            tree_method=params.get("tree_method", "auto"),
            predictor=params.get("predictor", "auto"),
            gpu_id=params.get("gpu_id", None)
        )
    elif classifier == "LogisticRegression":
        if GPU_AVAILABLE and use_gpu:
            base_model = cuLogisticRegression(
                max_iter= params.get("max_iter", 100),
                random_state=42,
                C= params.get("C", 1),
                tol= params.get("tol", 0.0001),
            )
            if normalize:
                base_model = make_pipeline(cuStandardScaler(), base_model)
        else:
            base_model = LogisticRegression(
                max_iter= params.get("max_iter", 100),
                random_state=42,
                C= params.get("C", 1),
                tol= params.get("tol", 0.0001),
            )
            if normalize:
                base_model = make_pipeline(StandardScaler(with_mean=False), base_model)
    elif classifier == "SVM":
        if GPU_AVAILABLE and use_gpu:
            base_model = cuSVC(
                max_iter=params.get("max_iter", 100),
                C=params.get("C", 1),
                probability=params.get("probability", False)
            )
            if normalize:
                base_model = make_pipeline(cuStandardScaler(), base_model)
        else:
            base_model = SVC(
                random_state=params.get("random_state", 42),
                max_iter=params.get("max_iter", 100),
                C=params.get("C", 1),
                probability=params.get("probability", False),
                tol=params.get("tol",0.001),
                kernel= params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale" if len(labels) > 2 else "auto")
            )
            if normalize:
                base_model = make_pipeline(StandardScaler(with_mean=False), base_model)

    # Handle grid search
    grid = None
    if grid_search:
        print("Performing Grid Search...\n")
        if classifier not in param_grids:
            raise ValueError(f"Grid search not supported for {classifier}")
            
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[classifier],
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=1 if GPU_AVAILABLE else -1  # Avoid GPU memory issues with parallel jobs
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print("Best hyperparameters:\n", grid.best_params_)
        
        # Update params with best parameters from grid search
        params.update(grid.best_params_)
    else:
        model = base_model
        if classifier == "XGBClassifier":
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        else:
            model.fit(X_train, y_train)

    # Evaluation
    # Training evaluation
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy: {train_accuracy}")
    print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
    save_classification_report(
        y_train, y_pred_train, train_accuracy, classifier, params,
        classification_report_path.replace("classification_report", "train_classification_report"),
        grid, None
    )

    # Validation evaluation
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        print(f"\nValidation Accuracy: {val_accuracy}")
        print("Validation Classification Report:\n", classification_report(y_val, y_pred_val))
        
        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(X_val)
            roc_auc_score = plot_multiclass_roc(
                y_val, y_val_prob, n_classes=len(labels),
                dataset_label='Validation', path=chart_path
            )
            print("Validation AUC-ROC Score:\n", roc_auc_score)
        
        save_classification_report(
            y_val, y_pred_val, val_accuracy, classifier, params,
            classification_report_path, grid, roc_auc_score
        )

    if classifier == "XGBClassifier":
        visualize_model(model, chart_path)
        
    save_model(model, save_path)
    print(f"Trained model saved to {save_path}\n\n")

# === Main Function ===
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a model using a specific embedding method.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file.")

    args = parser.parse_args()

    # Load the YAML configuration file
    config_path = config_path = args.config

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_params = config["model"]["params"]

    label_column = config["dataset"]["label_column"]
    labels = config["dataset"]["labels"]
    embedding_type = config["embeddings"]["type"]

    # Extract dataset configuration from YAML
    dataset_name = config["dataset"]["name"]
    dataset_train = config["dataset"]["train"]
    dataset_test = config["dataset"]["test"]
    dataset_val = config["dataset"]["val"] if "val" in config["dataset"] else dataset_test
    store_type = config["embeddings"]["store_type"] if "store_type" in config["embeddings"] else "csv"
    print("store_type:", store_type)
    
    X_train, y_train = load_embeddings(embedding_type=embedding_type, dataset_name=dataset_train, label_column=label_column,store_type=store_type)
    
    X_val, y_val = load_embeddings(embedding_type=embedding_type, dataset_name=dataset_val, label_column=label_column,store_type=store_type)

        
    # Train and evaluate the model
    output_path = f"{config['output']['model_save_path']}/{config['experiment_name']}_{embedding_type}_{dataset_name}_model.pkl"
    classifier = config["model"]["classifier"] if "classifier" in config["model"] else "XGBClassifier"
    timestamp = int(time.time() * 1000)
    classification_report_path = f"{config['output']['results_path']}/{config['experiment_name']}_classification_report_{timestamp}.json"
    if not os.path.exists(config['output']['results_path']):
        os.makedirs(config['output']['results_path'])
    chart_path = f"{config['output']['results_path']}/{config['experiment_name']}_learning_curve_{timestamp}.png"
    train_and_evaluate(X_train, y_train, X_val, y_val, model_params,
                               output_path, classification_report_path, chart_path, classifier=classifier,labels=labels)


if __name__ == "__main__":
    main()
