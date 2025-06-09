# src/embeddings/tfidf_embedding.py
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import yaml
from base_embedding import load_model, save_embeddings, save_model
import pandas as pd
import json
from utils import *
from scipy.sparse import csr_matrix


def generate_tfidf(corpus, max_features=5000, ngram_range=(1, 2), min_df=None, max_df=None, stats_path=None):
    """
    Generate TF-IDF embeddings for the given text corpus.

    Args:
        corpus (list): List of text documents.
        max_features (int): Maximum number of features to extract (vocabulary size).
        ngram_range (tuple): The range of n-grams to consider (e.g., (1, 2) for unigrams and bigrams).

    Returns:
        array: TF-IDF feature matrix.
        vectorizer: Fitted TfidfVectorizer model.
    """
    print(
        f"Training TF-IDF vectorizer with max_features={max_features} and ngram_range={ngram_range}...")

    start_time = time.time()
    initial_memory = get_memory_usage()
    print("Max features: ", max_features)
    print("Ngram range: ", ngram_range)
    
    vectorizer = TfidfVectorizer(max_features = max_features, ngram_range = ngram_range)

    tfidf_matrix = vectorizer.fit_transform(corpus)
    normalizer = Normalizer()
    tfidf_matrix = normalizer.fit_transform(tfidf_matrix)

    end_time = time.time()
    final_memory = get_memory_usage()
    training_time = end_time - start_time
    memory_used = final_memory - initial_memory

    # Collect training statistics
    training_stats = {
        "training_time_seconds": training_time,
        "memory_usage_mb": memory_used,
        "documents_per_second": len(corpus)/(training_time),
        "model_parameters": {
            "max_features": max_features,
            "ngram_range": ngram_range,

        },
        "dataset_stats": {
            "num_documents": len(corpus),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"\nModel Training Statistics:")
    print(f"Total time: {training_stats['training_time_seconds']:.2f} seconds")
    print(f"Memory usage: {training_stats['memory_usage_mb']:.2f} MB")
    print(
        f"Processing speed: {training_stats['documents_per_second']:.2f} documents per second")
    save_embedding_stats(training_stats, stats_path)
    return tfidf_matrix, vectorizer


def generate_from_vectorizer(corpus, vectorizer: TfidfVectorizer, stats_path=None):
    """
    Generate TF-IDF embeddings for the given text corpus using a pre-trained vectorizer.

    Args:
        corpus (list): List of text documents.
        vectorizer: Fitted TfidfVectorizer model.

    Returns:
        array: TF-IDF feature matrix.
    """
    print("Generating TF-IDF embeddings...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    total_samples = len(data)

    SparseMatrix = vectorizer.transform(corpus)

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

    return SparseMatrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TF-IDF embeddings.")
    parser.add_argument("--input_path", type=str, required=False,
                        help="Path to the input dataset (CSV).", default=None)
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the embeddings (CSV).")
    parser.add_argument("--model_save_path", type=str,
                        required=True, help="Path to save TFIDF model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment configuration file.")
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
        


    data = pd.read_csv(input_path)
    ngram_range = None

    if "ngram_range" in config["embeddings"]["params"]:
        ngram_range = tuple(
            map(int, config["embeddings"]["params"]["ngram_range"].split(",")))

    pre_process_params = config["embeddings"]["pre_process"] if "pre_process" in config["embeddings"] else None
    store_type = config["embeddings"]["store_type"] if "store_type" in config["embeddings"] else "csv"

    if pre_process_params:
        print("Preprocessing text...")
        data["text"] = data["text"].apply(partial(preprocess_text, **pre_process_params)) 
        
    corpus = data["text"].tolist()
    # Generate TF-IDF embeddings
    vectorizer = None
    data["embeddings"] = None
    if args.mode in ["encode", "val"]:
        vectorizer = load_model(args.model_save_path, model_type="tfidf")
        embeddings = generate_from_vectorizer(
            corpus, vectorizer, stats_path=stats_path)
    else:
        print("Training TF-IDF model...")
        embeddings, vectorizer = generate_tfidf(
            corpus,
            max_features=config["embeddings"]["params"]["max_features"] if "max_features" in config["embeddings"]["params"] else None,
            min_df=config["embeddings"]["params"]["min_df"] if "min_df" in config["embeddings"]["params"] else 1,
            max_df=config["embeddings"]["params"]["max_df"] if "max_df" in config["embeddings"]["params"] else 1,
            ngram_range=ngram_range,
            stats_path=stats_path)

    if store_type == "csv":
        data["embeddings"] = embeddings.toarray().tolist()
        data["embeddings"] = data["embeddings"].apply(lambda x: json.dumps(x))
        save_embeddings(data, args.output_path, store_type=store_type)
        
    elif store_type == "npz":
        labels = data["label"].to_numpy()
        save_embeddings({"X":embeddings,"y":labels}, args.output_path, store_type=store_type)
        # Hatanın kaynağı tf-idf vektörü üretmi, neden oluyor araştırmak lazım ama baseline için bu yeterli


    # Save embeddings
    if args.mode == "train":
        save_model(vectorizer, args.model_save_path)
