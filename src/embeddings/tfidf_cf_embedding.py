# src/embeddings/tfidf_cf_embedding.py
import sys
sys.path.append("../../")  # Adjust path to import from src
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from src.embeddings.base_embedding import load_model, save_embeddings, save_model
    from src.embeddings.utils import get_memory_usage, save_embedding_stats,preprocess_text
except ImportError:
    from base_embedding import load_model, save_embeddings, save_model
    from utils import get_memory_usage, save_embedding_stats, preprocess_text

import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
import argparse
import yaml
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from threading import Lock


def compute_category_weights(categories):
    """
    Compute category weights (category frequencies).

    Args:
        categories (list or array): List of category labels (e.g., class1, class2, etc.).

    Returns:
        dict: Dictionary containing category weights normalized by the category count.
    """
    category_freq = pd.Series(categories).value_counts(
        normalize=True)  # Normalize frequencies
    return category_freq.to_dict()

# Global variables for worker processes
_worker_labels = None
_worker_vectorizer = None
_worker_matrix = None

def init_worker(labels, vectorizer, matrix):
    global _worker_labels, _worker_vectorizer, _worker_matrix
    _worker_labels = labels
    _worker_vectorizer = vectorizer
    _worker_matrix = matrix

def process_chunk(chunk):
    chunk_class_freq = defaultdict(lambda: defaultdict(int))
    for doc_idx in chunk:
        label = _worker_labels[doc_idx]
        terms = _worker_vectorizer.inverse_transform(
            _worker_matrix[doc_idx].reshape(1, -1))[0]
        for term in terms:
            chunk_class_freq[label][term] += 1
    return chunk_class_freq


class Tf_Idf_Cf_Vectorizer:
    def __init__(self, tfidf_vectorizer: TfidfVectorizer, cf_weights: dict) -> None:
        self.tfidf_vectorizer = tfidf_vectorizer
        self.cf_weights = cf_weights

    @staticmethod
    def save(model, model_save_path: str, setting_save_path: str) -> None:
        if not model_save_path.endswith('.model'):
            model_save_path = model_save_path + '.model'
        save_model(model.tfidf_vectorizer, model_save_path)

        if not setting_save_path.endswith('.json'):
            setting_save_path = setting_save_path + '.json'
        with open(setting_save_path, 'w') as f:
            json.dump({
                "cf_weights": model.cf_weights
            }, f)

        if model.training_stats:
            save_embedding_stats(model.training_stats, model_save_path.replace(
                ".model", "training_stats.json"))

    @staticmethod
    def load_model(model_save_path: str, setting_save_path: str):
        if not model_save_path.endswith('.model'):
            model_save_path = model_save_path + '.model'
        if not setting_save_path.endswith('.json'):
            setting_save_path = setting_save_path + '.json'

        tfidf_vectorizer = load_model(model_save_path)
        with open(setting_save_path, 'r') as f:
            settings = json.load(f)

        cf_weights = defaultdict(lambda: defaultdict(int))
        for key in settings["cf_weights"]:
            cf_weights[int(key)] = settings["cf_weights"][key]
        print("Model loaded successfully.")
        return Tf_Idf_Cf_Vectorizer(tfidf_vectorizer, cf_weights)

    @staticmethod
    def compute_category_frequency(labels: list, tfidf_vectorizer: TfidfVectorizer, 
                                  tfidf_matrix: np.ndarray, max_workers: int = None):
        """
        Compute Category Frequency (CF) for all terms in TF-IDF vectorizer using multi-threading.
        
        Args:
            labels: List of document labels
            tfidf_vectorizer: Fitted TfidfVectorizer
            tfidf_matrix: TF-IDF matrix
            max_workers: Number of threads to use (None for automatic)
        """
        print("Computing category frequencies (multi-threaded)...")
        start_time = time.time()
        initial_memory = get_memory_usage()

        # We'll use a lock for thread-safe updates to the shared dictionary
        lock = Lock()
        class_freq = defaultdict(lambda: defaultdict(int))
        class_counts = Counter(labels)

        def process_document(doc_idx):
            nonlocal class_freq
            label = labels[doc_idx]
            terms = tfidf_vectorizer.inverse_transform(
                tfidf_matrix[doc_idx].reshape(1, -1))[0]
            
            # Acquire lock before updating shared dictionary
            with lock:
                for term in terms:
                    class_freq[label][term] += 1

        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_document, range(len(labels))),
                 total=len(labels), desc="Computing CF"))

        # Normalize CF by class size
        for label in class_freq:
            total_docs_in_class = class_counts.get(label, 1)
            for term in class_freq[label]:
                class_freq[label][term] /= total_docs_in_class

        end_time = time.time()
        final_memory = get_memory_usage()

        print(f"CF computation completed in {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {final_memory - initial_memory:.2f} MB")

        return class_freq

    @staticmethod
    def compute_tf_idf_cf_matrix(vectorizer: TfidfVectorizer, tfidf_matrix: np.array, labels: List[str], class_freq: dict) -> csr_matrix:
        """
        Compute the adjusted TF-IDF-CF matrix.

        Args:
            tfidf_matrix (scipy.sparse.csr_matrix): Initial TF-IDF matrix.
            terms (list): List of terms as per TfidfVectorizer.
            cf_weights (dict): Category Frequency (CF) weight for each term.

        Returns:
            np.ndarray: Adjusted TF-IDF-CF matrix.
        """
        start_time = time.time()
        initial_memory = get_memory_usage()

        tfidf_cf_matrix = np.zeros_like(tfidf_matrix.toarray())
        default_cf = 1e-6  # Small default value for unseen terms
        for doc_idx, label in tqdm(enumerate(labels), total=len(labels), desc="Computing TF-IDF-CF matrix"):
            terms = vectorizer.inverse_transform(tfidf_matrix[doc_idx])[0]
            for term in terms:
                term_idx = vectorizer.vocabulary_[term]
                cf_score = class_freq[label].get(term, default_cf)
                tfidf_cf_matrix[doc_idx,
                                term_idx] = tfidf_matrix[doc_idx, term_idx] * cf_score

        end_time = time.time()
        final_memory = get_memory_usage()

        print(
            f"Matrix computation completed in {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {final_memory - initial_memory:.2f} MB")
        # Convert the dense matrix back to sparse CSR format for efficiency
        tfidf_cf_matrix = csr_matrix(tfidf_cf_matrix)
        print(f"Converted result to CSR matrix with shape {tfidf_cf_matrix.shape}")
        print(f"Matrix density: {tfidf_cf_matrix.nnz / (tfidf_cf_matrix.shape[0] * tfidf_cf_matrix.shape[1]):.6f}")

        return tfidf_cf_matrix

    @staticmethod
    def build_tf_idf_cf(df_sample: pd.DataFrame, category_column: str = 'category', params: dict = {}):
        default_params: dict = {
                                #"max_df": 0.85,  # Ignore terms that appear in more than 50% of the documents
                                #"min_df": 0.01,  # Ignore terms that appear in fewer than 2 documents
                                "stop_words": 'english',
                                "max_features": None,
                                "ngram_range": (1, 1)}
        
        override_params = {**default_params, **params}
        print("Parameters:", override_params)
        X = df_sample['text']
        start_time = time.time()
        initial_memory = get_memory_usage()

        class_freq = defaultdict(lambda: defaultdict(int))
        labels = df_sample[category_column].to_list()

        print("Training TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(**override_params)
        tfidf_matrix = tfidf_vectorizer.fit_transform(X)    # Sparse TF-IDF matrix

        # Compute Category Frequency (CF)
        cf_weights = Tf_Idf_Cf_Vectorizer.compute_category_frequency(
            labels, tfidf_vectorizer, tfidf_matrix)
        model = Tf_Idf_Cf_Vectorizer(tfidf_vectorizer, cf_weights)

        end_time = time.time()
        final_memory = get_memory_usage()

        training_stats = {
            "training_time_seconds": end_time - start_time,
            "memory_usage_mb": final_memory - initial_memory,
            "documents_per_second": len(df_sample)/(end_time - start_time),
            "model_parameters": {
                **override_params,
                "num_categories": len(labels),
            },
            "dataset_stats": {
                "num_documents": len(df_sample),
                "num_categories": len(labels),
                "category_distribution": compute_category_weights(df_sample[category_column])
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\nModel Training Statistics:")
        print(
            f"Total time: {training_stats['training_time_seconds']:.2f} seconds")
        print(f"Memory usage: {training_stats['memory_usage_mb']:.2f} MB")
        print(
            f"Processing speed: {training_stats['documents_per_second']:.2f} documents per second")

        model.training_stats = training_stats
        return model

    def compute_tfidfcf_embedding(self, data: pd.DataFrame, text_column: str = 'text', category_column="label", stats_path: str = None) -> csr_matrix:

        print("Generating TF-IDF-CF embeddings...")
        start_time = time.time()
        initial_memory = get_memory_usage()
        total_samples = len(data)

        tf_idf_matrix = self.tfidf_vectorizer.transform(data[text_column])
        labels = data[category_column].values
        tfidf_cf_matrix = Tf_Idf_Cf_Vectorizer.compute_tf_idf_cf_matrix(self.tfidf_vectorizer, tf_idf_matrix, labels, self.cf_weights)
        normalizer = Normalizer()
        tfidf_cf_matrix = normalizer.fit_transform(tfidf_cf_matrix)

    # Calculate performance metrics
        end_time = time.time()
        final_memory = get_memory_usage()

        processing_time = end_time - start_time
        memory_used = final_memory - initial_memory
        successful_samples = total_samples

        print(f"\nEmbedding Generation Statistics:")
        print(f"Total time: {processing_time:.2f} seconds")
        print(
            f"Processing speed: {successful_samples/processing_time:.2f} samples per second")
        print(f"Memory usage: {memory_used:.2f} MB")
        print(f"Successful samples: {successful_samples}/{total_samples}")
        print(f"Error rate: {(0/total_samples)*100:.2f}%")
        embedding_generation_stats = {
            "processing_time_seconds": processing_time,
            "memory_usage_mb": memory_used,
            "successful_samples": successful_samples,
            "total_samples": total_samples,
            "error_rate": 0/total_samples,
            "samples_per_second": successful_samples/processing_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_embedding_stats(embedding_generation_stats, stats_path)
        return tfidf_cf_matrix


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Generate TF-IDF-CF embeddings.")
    parser.add_argument("--input_path", type=str, required=False,
                        help="Path to the input dataset (CSV).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the embeddings (CSV).")
    parser.add_argument("--model_save_path", type=str,
                        required=True, help="Path to save TFIDFCF model.")
    parser.add_argument("--setting_save_path", type=str,
                        required=True, help="Path to save TFIDFCF settings.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    parser.add_argument("--mode", default="train", type=str,
                        required=True, help="Mode of operation: train or encode.")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
        
    stats_path:str = config["output"]["results_path"] if "results_path" in config["output"] else None
    if stats_path is not None:
        if not stats_path.endswith("/"):
            stats_path += f"/embedding_stats_{args.mode}.json"
        else:
            stats_path += f"embedding_stats_{args.mode}.json"
    else:
        stats_path = args.output_path.replace(".csv", f"_{args.mode}_stats.json")

    if args.input_path is None and args.mode == "train":
        input_path = f"data/processed/{config['dataset']['train']}.csv"
    elif args.input_path is None and args.mode == "encode":
        input_path = f"data/processed/{config['dataset']['test']}.csv"
    elif args.input_path is None and args.mode == "val":
        input_path = f"data/processed/{config['dataset']['val']}.csv"
    else:
        input_path = args.input_path

    category_column = config["dataset"]["label_column"]

    # Load dataset
    data = pd.read_csv(input_path)
    print(f"Loading dataset from {input_path}")
    
    pre_process_params = config["embeddings"]["pre_process"] if "pre_process" in config["embeddings"] else None
    if pre_process_params:
        pre_process_fun = partial(preprocess_text, **pre_process_params)
        print("Pre processing..")
        data["text"] = data["text"].apply(pre_process_fun)
        print("Pre process completed")

    ngram_range = None
    if "ngram_range" in config["embeddings"]["params"]:
        ngram_range = tuple(
            map(int, config["embeddings"]["params"]["ngram_range"].split(",")))

    params = config["embeddings"]["params"]
    params["ngram_range"] = ngram_range
    store_type = config["embeddings"]["store_type"] if "store_type" in config["embeddings"] else "csv"

    # Generate TF-IDF-CF embeddings
    if args.mode in ["encode","val"]:
        model = Tf_Idf_Cf_Vectorizer.load_model(
            args.model_save_path, args.setting_save_path)
    else:
        if 'pre_process' in params: 
            del params['pre_process']  # Remove pre_process from params
        model = Tf_Idf_Cf_Vectorizer.build_tf_idf_cf(data, category_column,  params)

    matrix = model.compute_tfidfcf_embedding(data, category_column=category_column, stats_path=stats_path)
    if store_type == "csv":
        data = data.drop(columns=["text"])
        data["embeddings"] = matrix.toarray().tolist()
        data["embeddings"] = data["embeddings"].apply(lambda x: json.dumps(x))
        save_embeddings(data, args.output_path)
        
    elif store_type == "npz":
        labels = data["label"].to_numpy()
        save_embeddings({"X":matrix,"y":labels}, args.output_path, store_type=store_type)
        
    # Save model and settings
    if args.mode == "train":
        print("Saving model and settings...")
        Tf_Idf_Cf_Vectorizer.save(
            model, args.model_save_path, args.setting_save_path)
    print("Done!")
