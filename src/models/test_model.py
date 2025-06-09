import argparse
import time
import yaml
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from model_utils import load_model, save_model
from sklearn.preprocessing import LabelEncoder 
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
sys.path.append("./src/")
from embeddings.base_embedding import load_embeddings as loader
import os,json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_classification_report(report_path, output_path=None):
    """
    Create visual representations of the classification report using both heatmap and bar chart.
    
    Args:
        report_path (str): Path to the JSON file containing the classification report
        output_path (str): Path where the visualization should be saved. If None, displays the plot.
    """
    # Load the classification report from JSON
    with open(report_path, 'r') as f:
        report_dict = json.load(f)
    
    # Extract relevant metrics and classes
    classes = [key for key in report_dict.keys() if key not in ['accuracy', 'macro avg', 'weighted avg', 'model']]
    metrics = ['precision', 'recall', 'f1-score']
    
    # Create a matrix for the visualizations
    data = []
    for class_name in classes:
        row = [report_dict[class_name][metric] for metric in metrics]
        data.append(row)
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 8))
    
    # Heatmap subplot
    plt.subplot(1, 2, 1)
    sns.heatmap(data_array, 
                annot=True, 
                cmap='YlOrRd', 
                xticklabels=metrics, 
                yticklabels=classes, 
                fmt='.3f',
                vmin=0, 
                vmax=1)
    
    plt.title('Classification Report Heatmap')
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    
    # Bar chart subplot
    plt.subplot(1, 2, 2)
    x = np.arange(len(classes))
    width = 0.25
    
    # Plot bars for each metric
    plt.bar(x - width, data_array[:, 0], width, label='Precision', color='skyblue')
    plt.bar(x, data_array[:, 1], width, label='Recall', color='lightgreen')
    plt.bar(x + width, data_array[:, 2], width, label='F1-score', color='salmon')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Metrics by Class')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add overall accuracy as text
    plt.figtext(0.99, 0.01, f'Overall Accuracy: {report_dict["accuracy"]:.3f}', 
                horizontalalignment='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Classification report visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

# === Embedding Loaders ===
def load_embeddings(path=None,embedding_type="tfidf", dataset_name="ds.csv", embeddings_dir="data/embeddings",label_column="label",store_type="csv"):
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
        embedding_file = f"{embeddings_dir}/{embedding_type}_{dataset_name}_embeddings.csv"
    if not os.path.exists(embedding_file.replace(".csv", "."+store_type)):
        raise FileNotFoundError(f"Embedding file not found at {embedding_file}. Make sure to run the generate_embeddings.sh script first.")
    print(f"Loading embeddings from {embedding_file}...")
    return loader(embedding_file,label_column=label_column,store_type=store_type)

def visualize_confusion_matrix(y,y_pred,target_names,chart_path=None,classifier="XGBClassifier"):
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')
    plt.title(f"{classifier} Confusion Matrix")
    if chart_path:
        plt.savefig(chart_path)
        print(f"Confusion Matrix saved to {chart_path}")
        
# === Model Training and Evaluation ===
def load_and_evaluate_model(X_test, y_test, save_path,classification_report_path,chart_path=None,classifier="XGBoostClassifier",categories=None):
    """
    Train an XGBoost classifier and evaluate on the test set.

    Args:
        X_train (array-like): Training dataset features.
        y_train (array-like): Training dataset labels.
        X_test (array-like): Testing dataset features.
        y_test (array-like): Testing dataset labels.
        save_path (str): File path to save the trained model.

    Returns:
        None
    """
    print(f"Testing {classifier}")
    model = load_model(save_path)
    # Evaluate on the test set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing Accuracy: {accuracy}")
    print("Testing Classification Report:\n", classification_report(y_test, y_pred))

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_dict["model"]=classifier
    
    # Save the classification report to a JSON file
    with open(classification_report_path, "w") as f:
        json.dump(report_dict, f, indent=4)  # Save with indentation for readability
        
    print(f"Testing Classification report saved to {classification_report_path}")
    
    # Generate visualization of the classification report
    viz_output_path = classification_report_path.replace('.json', '_viz.png')
    visualize_classification_report(classification_report_path, viz_output_path)
    visualize_confusion_matrix(y_test, y_pred, categories, chart_path=chart_path,classifier=classifier)
    
def predict(X_test, model_path):
    """
    Predict labels for the test set using a pre-trained model.

    Args:
        X_test (array-like): Test dataset features.
        model_path (str): Path to the pre-trained model.

    Returns:
        array-like: Predicted labels.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    return y_pred

def preadict_and_save(X_test, model_path, output_path):
    """
    Predict labels for the test set and save the predictions to a CSV file.

    Args:
        X_test (array-like): Test dataset features.
        model_path (str): Path to the pre-trained model.
        output_path (str): Path to save the predictions.

    Returns:
        None
    """
    y_pred = predict(X_test, model_path)
    df = pd.DataFrame(y_pred, columns=["predicted_label"])
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# === Main Function ===
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model using a specific embedding method.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--mode", type=str, required=False, help="Mode of operation: evaluate or predict. Default is 'evaluate'.", default="evaluate")
    
    args = parser.parse_args()
    mode = args.mode.lower()
    if mode not in ["evaluate", "predict"]:
        raise ValueError("Invalid mode. Use 'evaluate' or 'predict'.")

    # Load the YAML configuration file
    config_path =args.config
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    label_column = config["dataset"]["label_column"]
    embedding_type = config["embeddings"]["type"]

    # Extract dataset configuration from YAML
    dataset_name = config["dataset"]["name"]
    dataset = config["dataset"]["test"]
    categories= config["dataset"]["categories"] if "categories" in config["dataset"] else None
    classifier = config["model"]["classifier"] if "classifier" in config["model"] else "XGBoostClassifier"
    store_type = config["embeddings"]["store_type"] if "store_type" in config["embeddings"] else "csv"
    X_test, y_test = load_embeddings(embedding_type=embedding_type, dataset_name=dataset,label_column=label_column,store_type=store_type)

    # Train and evaluate the model
    timestamp = int(time.time() * 1000)
    model_path = f"{config['output']['model_save_path']}/{config['experiment_name']}_{embedding_type}_{dataset_name}_model.pkl"
    classification_report_path = f"{config['output']['results_path']}/eval_classification_report_{timestamp}.json"
    chart_path = f"{config['output']['results_path']}/eval_confusion_matrix_{timestamp}.png"
    load_and_evaluate_model( X_test, y_test, model_path,classification_report_path,chart_path,classifier=classifier,categories=categories)
    
if __name__ == "__main__":
    main()
