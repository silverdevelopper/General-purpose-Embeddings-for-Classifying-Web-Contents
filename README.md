# General-purpose Embedding for Classifying Web Content

## **Overview**
This project is a modular framework for text classification tasks. It supports multiple datasets and embedding methods (e.g., **Doc2Vec, FastText, HuggingFace, TF-IDF, and TF-IDF-CF**) and allows users to configure and design their own experiments via YAML configuration files. The workflow includes preprocessing, generating embeddings, splitting datasets, training models, and evaluating predictions, all integrated into an easily extendable structure.

---

## **Workflow Overview**

Here's the workflow:

1. **Split Dataset**: Split the dataset into training and testing subsets.
2. **Preprocess Dataset**: Clean and normalize the input text data.
3. **Generate Embeddings**: Compute embeddings for both training and test datasets.
4. **Train Model**: Train a machine learning model using the precomputed embeddings.
5. **Evaluate Model**: Evaluate the model's performance using accuracy, F1-score, and confusion matrix.

---

## **Project Structure**

```
project_name/
├── config-examples/           # Example YAML configurations
│   ├── wiki-cde.yaml
│   ├── wiki-doc2vec-100.yaml
│   └── ...
│
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Preprocessed datasets
│   ├── embeddings/           # Precomputed embeddings
│   ├── splits/               # Train/test splits
│   ├── preprocess.py         # Preprocessing script
│   ├── split_dataset.py      # Dataset splitting utility
│   └── loader.py            # Dataset loading utilities
│
├── experiments/              # Experiment configurations
│   ├── 20news-bert.yaml
│   ├── 20news-cde.yaml
│   └── ...
│
├── notebooks/               # Jupyter notebooks for analysis
│   ├── doc2vec.ipynb
│   ├── fasttext.ipynb
│   ├── hf.ipynb
│   └── ...
│
├── src/
│   ├── embeddings/          # Embedding implementations
│   │   ├── base_embedding.py
│   │   ├── cde_small_v1_embedding.py
│   │   ├── doc2vec_embedding.py
│   │   ├── fasttext_embedding.py
│   │   ├── hf_embedding.py
│   │   ├── tfidf_embedding.py
│   │   └── tfidf_cf_embedding.py
│   │
│   ├── evaluations/         # Evaluation utilities
│   │   ├── confusion_matrix.py
│   │   ├── metrics.py
│   │   ├── results_aggregator.py
│   │   └── visualize_results.py
│   │
│   └── models/             # Model training and testing
│       ├── train_model.py
│       ├── test_model.py
│       ├── model_utils.py
│       └── hyperparameter_tuning.py
│
├── scripts/                # Automation scripts
│   ├── evaluate.sh
│   ├── generate_embeddings.sh
│   ├── preprocess.sh
│   ├── split_dataset.sh
│   └── train.sh
│
├── results/               # Evaluation results and metrics
└── models/               # Saved models
```

---

## **Features**

- **Multiple Embedding Methods**:
  - **Doc2Vec**: Dense paragraph vector embeddings
  - **FastText**: Subword-based embeddings
  - **TF-IDF**: Term frequency-inverse document frequency
  - **TF-IDF-CF**: Category-weighted TF-IDF
  - **HuggingFace**: Pre-trained transformer models (BERT, RoBERTa)
  - **CDE**: Custom Document Embeddings

- **Comprehensive Pipeline**:
  - Data preprocessing and cleaning
  - Embedding generation
  - Model training and evaluation
  - Results visualization

- **Experiment Management**:
  - YAML-based configuration
  - Separate configs for different datasets and embedding types
  - Easy parameter tuning

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix visualization
  - Results aggregation

---

## **Usage Guide**

### **1. Environment Setup**
```bash
pip install -r requirements.txt
```

### **2. Data Preparation**
Place your dataset in `data/raw/` and preprocess it:
```bash
bash scripts/preprocess.sh your_dataset
```

### **3. Generate Embeddings**
First train the embedder:
```bash
bash scripts/generate_embeddings.sh --mode train experiments/your-config.yaml
```

Then generate embeddings for test data:
```bash
bash scripts/generate_embeddings.sh --mode encode experiments/your-config.yaml
```

### **4. Train Model**
```bash
bash scripts/train.sh experiments/your-config.yaml
```

### **5. Evaluate Results**
```bash
bash scripts/evaluate.sh experiments/your-config.yaml
```

---

## **Configuration Format**

Example configuration file (`experiments/20news-fasttext-1000.yaml`):

```yaml
experiment_name: "fasttext_20news_1000"

dataset:
  name: "20_newsgroups"
  train: "20_newsgroups_train"
  test: "20_newsgroups_test"
  label_column: "label"

embeddings:
  type: "fasttext"
  params:
    vector_size: 1000
    window: 8
    min_count: 3
    epochs: 20

model:
  classifier: "xgboost"
  params:
    learning_rate: 0.1
    max_depth: 3
    n_estimators: 400
    num_class: 20
    eval_metric: "mlogloss"
    objective: "multi:softmax"

output:
  embeddings_path: "data/embeddings"
  model_save_path: "models/"
  results_path: "results"
```

---

## **Notebooks**

The `notebooks/` directory contains Jupyter notebooks for:
- Embedding analysis and visualization
- Model experimentation
- Results analysis
- Data exploration

Key notebooks:
- `doc2vec.ipynb`: Doc2Vec embedding analysis
- `fasttext.ipynb`: FastText model experiments
- `hf.ipynb`: HuggingFace transformers usage
- `RoBERTa.ipynb`: RoBERTa model experiments

---

## **Results**

Evaluation results are stored in the `results/` directory:
- Classification reports (JSON)
- Learning curves (PNG)
- Confusion matrices
- Aggregated metrics

---

## **Extending the Framework**

### **Adding New Embeddings**
1. Create a new class in `src/embeddings/`
2. Inherit from `BaseEmbedding`
3. Implement required methods
4. Update configuration files

### **Custom Models**
1. Add model implementation in `src/models/`
2. Update training and evaluation scripts
3. Create corresponding configuration templates

---

## **Contributing**

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.
