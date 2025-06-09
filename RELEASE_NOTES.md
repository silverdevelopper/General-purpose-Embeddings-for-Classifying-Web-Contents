# Release Notes - General-purpose Embedding for Classifying Web Content

## Version 1.0.0

### Key Features

#### 1. Modular Text Classification Framework
- Support for multiple embedding methods:
  - Doc2Vec (dense paragraph vector embeddings)
  - FastText (subword-based embeddings)
  - TF-IDF (term frequency-inverse document frequency)
  - TF-IDF-CF (category-weighted TF-IDF)
  - HuggingFace Transformers (BERT, RoBERTa)
  - CDE (Custom Document Embeddings)

#### 2. Comprehensive Data Pipeline
- Automated data preprocessing and cleaning
- Configurable dataset splitting
- Efficient embedding generation
- Model training and evaluation
- Results visualization and metrics reporting

#### 3. Experiment Management
- YAML-based configuration system
- Separate configs for different datasets and embedding types
- Flexible parameter tuning
- Example configurations provided for various use cases

#### 4. Performance Metrics
- Implementation of multiple evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix visualization
- Results aggregation and reporting

### Performance Highlights

#### TF-IDF Model Performance (20 Newsgroups Dataset)
- Overall Accuracy: 66.15%
- Macro Average:
  - Precision: 65.31%
  - Recall: 65.13%
  - F1-score: 64.90%
- Best performing categories achieved F1-scores above 80%
- Consistent performance across majority of categories

### Project Structure
- Organized directory structure for:
  - Configuration examples
  - Data management
  - Experiment configurations
  - Source code
  - Evaluation results
  - Analysis notebooks

### Documentation
- Comprehensive README with setup instructions
- Detailed usage guides
- Example configurations
- Jupyter notebooks for analysis and experimentation

### Supported Datasets
- 20 Newsgroups
- Wikipedia
- Support for custom datasets through modular design

### Development Tools
- Shell scripts for automation
- Jupyter notebooks for analysis
- Modular Python codebase
- Configuration-driven experimentation

### Known Issues & Future Work
- EUR-lex dataset integration pending
- Medical term dataset requires updates
- Wikipedia chunks need optimization

### Requirements
- Python dependencies listed in requirements.txt
- Support for various embedding methods and their dependencies

### Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Configure experiment using YAML
3. Preprocess data using provided scripts
4. Generate embeddings
5. Train and evaluate models

For detailed setup and usage instructions, refer to the README.md file.
