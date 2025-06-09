# Scripts Documentation

This directory contains shell scripts for automating various tasks in the machine learning pipeline. Each script is designed to handle specific aspects of data processing, model training, and evaluation.

## Available Scripts

1. [preprocess.sh](#preprocess)
2. [split_dataset.sh](#split-dataset)
3. [generate_embeddings.sh](#generate-embeddings)
4. [train.sh](#train)
5. [evaluate.sh](#evaluate)
6. [run_experiment.sh](#run-experiment)

## Script Details

<a name="preprocess"></a>
### `preprocess.sh`
Preprocesses raw data by cleaning text and preparing it for model training.

```bash
bash scripts/preprocess.sh DATASET_NAME
```
- Input: `data/raw/DATASET_NAME.csv`
- Output: `data/processed/DATASET_NAME.csv`

<a name="split-dataset"></a>
### `split_dataset.sh`
Splits preprocessed data into training and testing sets.

```bash
bash scripts/split_dataset.sh DATASET_NAME
```
- Input: `data/processed/DATASET_NAME.csv`
- Output: 
  - `data/splits/DATASET_NAME_train.csv`
  - `data/splits/DATASET_NAME_test.csv`

<a name="generate-embeddings"></a>
### `generate_embeddings.sh`
Generates text embeddings using various methods (Doc2Vec, FastText, TF-IDF, etc.).

```bash
bash scripts/generate_embeddings.sh DATASET_NAME [--doc2vec] [--fasttext] [--tfidf] [--tfidf-cf] [--huggingface]
```

Options:
- `--doc2vec`: Generate Doc2Vec embeddings
- `--fasttext`: Generate FastText embeddings
- `--tfidf`: Generate TF-IDF embeddings
- `--tfidf-cf`: Generate TF-IDF with category frequency embeddings
- `--huggingface`: Generate HuggingFace transformer embeddings

Outputs:
- Models: `models/[embedding_type]_DATASET_NAME.model`
- Embeddings: `data/embeddings/[embedding_type]_DATASET_NAME_embeddings.csv`
- Training Statistics: `models/[embedding_type]_DATASET_NAME_training_stats.json`

<a name="train"></a>
### `train.sh`
Trains a model using specified embeddings and configuration.

```bash
bash scripts/train.sh DATASET_NAME CONFIG_FILE EMBEDDING_TYPE
```
Example:
```bash
bash scripts/train.sh wikipedia_xs configs/wiki-fasttext-100.yaml fasttext
```

<a name="evaluate"></a>
### `evaluate.sh`
Evaluates model performance and generates metrics.

```bash
bash scripts/evaluate.sh DATASET_NAME
```
Outputs:
- Metrics: `results/DATASET_NAME_evaluation.json`
- Visualizations: `results/DATASET_NAME_evaluation_classification_report_viz.png`

<a name="run-experiment"></a>
### `run_experiment.sh`
Executes a complete experiment pipeline from preprocessing to evaluation.

```bash
bash scripts/run_experiment.sh CONFIG_FILE
```

This script:
1. Reads experiment configuration from the YAML file
2. Preprocesses the dataset if needed
3. Generates embeddings based on the specified method
4. Trains the model
5. Evaluates and saves results

Example:
```bash
bash scripts/run_experiment.sh experiments/patent-xs-cde.yaml
```

## Directory Structure

```
project/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed datasets
│   ├── splits/           # Train/test splits
│   └── embeddings/       # Generated embeddings
├── models/               # Saved models and training stats
├── results/              # Evaluation results
└── scripts/             # Shell scripts
```

## Configuration Files

Experiment configurations are defined in YAML files with the following structure:

```yaml
experiment_name: "patent_abstract_xs_tfidf-5000"

dataset:
  name: "patent_xs_abstract"
  train: "patent_xs_abstract_train"
  test: "patent_xs_abstract_test"
  label_column: "label"

embeddings:
  type: "tfidf"  # Options: doc2vec, fasttext, tfidf, tfidf_cf, huggingface
  params: 
    max_features: 5000
    ngram_range: "1,6"

model:
  classifier: "xgboost"  
  params:
    learning_rate: 0.5
    max_depth: 3
    n_estimators: 400
    num_class: 9
    eval_metric: "mlogloss" 
    objective: "multi:softmax"
output:
  embeddings_path: "data/embeddings"
  model_save_path: "models/"
  results_path: "results"
```

## Error Handling

Common issues and solutions:

1. **Missing Data**
   ```bash
   Error: Dataset not found at data/raw/DATASET_NAME.csv
   Solution: Ensure dataset exists in the correct location
   ```

2. **Invalid Configuration**
   ```bash
   Error: Missing required field in config file
   Solution: Check configuration file format
   ```

3. **Resource Issues**
   ```bash
   Error: Memory allocation failed
   Solution: Reduce batch size or dataset size
   ```

## Best Practices

1. **Data Organization**
   - Keep raw data in `data/raw/`
   - Use consistent naming conventions
   - Maintain separate train/test splits

2. **Configuration Management**
   - Store configurations in YAML files
   - Version control your configs
   - Document parameter choices

3. **Results Management**
   - Save all experiment results
   - Use meaningful filenames
   - Include timestamps in output files

## Dependencies

Ensure all required Python packages are installed:
```bash
pip install -r requirements.txt
```

Required packages include:
- pandas
- numpy
- scikit-learn
- gensim
- transformers
- tqdm
- matplotlib
- seaborn

## Contributing

When adding new scripts:
1. Follow existing naming conventions
2. Update this README
3. Include error handling
4. Add usage examples
