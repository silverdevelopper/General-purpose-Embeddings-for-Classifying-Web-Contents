## **Purpose**
The `src/models/` folder contains core scripts for managing machine learning models, including utilities for saving/loading models, evaluating models on test datasets, hyperparameter tuning, and generating predictions. These scripts are designed to provide modular and reusable functionality for different parts of the machine learning pipeline.

---

## **Folder Overview**
```
src/models/
├── train_model.py              # Train a model using a given embedding type and configuration
├── test_model.py               # Test a trained model and save predictions for evaluation
├── model_utils.py              # Utility functions for saving/loading models and making predictions
├── hyperparameter_tuning.py    # Perform hyperparameter tuning for a model
└── README.md                   # Documentation for the src/models/ folder
```

---

## **File Descriptions and Usage**

### **1. train_model.py**
The `train_model.py` script is the main training logic for training machine learning models (e.g., XGBoost) in the project. It allows you to train models using various embedding types (e.g., Doc2Vec, FastText, TF-IDF) and configurations specified in a YAML file. The script dynamically loads the dataset, generates embeddings, splits the dataset, and saves the trained model.

#### **Command**
```bash
python src/models/train_model.py --config PATH_TO_CONFIG --type EMBEDDING_TYPE
```

#### **Arguments**
- `--config`: Path to the YAML configuration file containing dataset paths, embedding parameters, model parameters, and other settings.
- `--type`: The type of embedding to use for training (`doc2vec`, `tfidf`, `fasttext`, `huggingface`, `tfidf_cf`).

#### **Example**
```bash
# Train a model using a doc2vec embedding
python src/models/train_model.py --config experiments/experiment_1.yaml --type doc2vec

# Train a model using TF-IDF
python src/models/train_model.py --config experiments/tfidf_experiment.yaml --type tfidf
```

#### **Output**
- Trained model saved to the path specified in the `config.yaml` file (e.g., in the `models/` directory).

---

### **2. test_model.py**
The `test_model.py` script is used to test a trained model on a given test dataset. It generates predictions for the test set and saves them to a `.csv` file for evaluation.

#### **Command**
```bash
python src/models/test_model.py --model_path MODEL_PATH --test_path TEST_DATA_PATH --output_path OUTPUT_PATH --text_column TEXT_COLUMN
```

#### **Arguments**
- `--model_path`: Path to the trained model file (e.g., `models/trained_model.pkl`).
- `--test_path`: Path to the test dataset (CSV file).
- `--output_path`: Path to save predictions (CSV file).
- `--text_column`: Name of the column containing text data in the test dataset (default: `text`).

#### **Example**
```bash
python src/models/test_model.py --model_path models/trained_model.pkl --test_path data/splits/test.csv --output_path results/predictions.csv
```

#### **Output**
- Predictions saved in the specified file (e.g., `results/predictions.csv`).

#### **Dependencies**
This script uses utility functions from `model_utils.py` to load the model and generate predictions.

---

### **3. model_utils.py**
The `model_utils.py` script contains utility functions for saving/loading models and generating predictions. These functions are designed to be reusable across other scripts in the project.

#### **Functions**
1. **`save_model(model, model_path)`**:
   - Saves a trained model to a file using `joblib`.
   - **Arguments**:
     - `model`: The trained model to save.
     - `model_path`: Path to save the model file (e.g., `.pkl`).
   - **Example**:
     ```python
     from src.models.model_utils import save_model
     save_model(model, "models/trained_model.pkl")
     ```

2. **`load_model(model_path)`**:
   - Loads a trained model from a file.
   - **Arguments**:
     - `model_path`: Path to the model file (e.g., `.pkl`).
   - **Example**:
     ```python
     from src.models.model_utils import load_model
     model = load_model("models/trained_model.pkl")
     ```

3. **`predict_model(model, texts)`**:
   - Generates predictions for text inputs using a trained model.
   - **Arguments**:
     - `model`: The trained model.
     - `texts`: List of input text data for prediction.
   - **Returns**: List of predictions.
   - **Example**:
     ```python
     from src.models.model_utils import predict_model
     predictions = predict_model(model, ["This is a test sentence.", "Another input for prediction."])
     ```

---

### **4. hyperparameter_tuning.py**
The `hyperparameter_tuning.py` script performs hyperparameter tuning for a model (e.g., XGBoost) using Grid Search or cross-validation and logs the best parameters for future experiments.

#### **Command**
```bash
python src/models/hyperparameter_tuning.py --embedding_path EMBEDDINGS_FILE --label_path LABELS_FILE --output_path OUTPUT_PATH
```

#### **Arguments**
- `--embedding_path`: Path to the file containing feature embeddings (e.g., `data/embeddings/tfidf_embeddings.csv`).
- `--label_path`: Path to the file containing label data (e.g., `data/processed/labels.csv`).
- `--output_path`: Path to save the best hyperparameter configuration (e.g., `results/hyperparam_tuning.json`).

#### **Example**
```bash
python src/models/hyperparameter_tuning.py --embedding_path data/embeddings/tfidf_embeddings.csv --label_path data/processed/labels.csv --output_path results/hyperparam_tuning.json
```

#### **Output**
- A JSON file containing the best hyperparameter configuration (e.g., `results/hyperparam_tuning.json`).

#### **Dependencies**
- Uses `GridSearchCV` from `scikit-learn` and is designed primarily for scikit-learn-compatible models like XGBoost.

---

## **Workflow Overview**
1. Use **`train_model.py`** to train your model using any embedding type and dataset configuration.
2. Use **`test_model.py`** to generate predictions from a trained model for the test dataset.
3. Use **`model_utils.py`** for utilities like saving/loading models and making predictions as reusable modular functions.
4. Use **`hyperparameter_tuning.py`** to optimize model hyperparameters and improve performance.

---

## **Folder Dependencies**
The files in the `src/models/` folder rely on:
1. **Feature Generators**:
   - Predefined embeddings (e.g., Doc2Vec, FastText) located in `data/embeddings/`.
2. **Dataset Splits**:
   - Train/test datasets from `data/splits/`.

---

## Example Workflow for Model Training and Testing
1. **Train a Model**:
   ```bash
   python src/models/train_model.py --config configs/experiment_1.yaml --type doc2vec
   ```
2. **Test the Model**:
   ```bash
   python src/models/test_model.py --model_path models/trained_model.pkl --test_path data/splits/test.csv --output_path results/predictions.csv
   ```
3. **Perform Hyperparameter Tuning**:
   ```bash
   python src/models/hyperparameter_tuning.py --embedding_path data/embeddings/tfidf_embeddings.csv --label_path data/processed/labels.csv --output_path results/hyperparam_tuning.json
   ```

---

## Notes
- Each script is modular and can be reused or extended for different classifiers or tasks.
- Ensure consistent file paths and column names in your datasets for compatibility with these scripts.
- Modify hyperparameter grids in `hyperparameter_tuning.py` as needed for your use case.