# src/embeddings/fasttext_embedding.py

from functools import partial
from typing import Union
from gensim.models import FastText
import numpy as np
from base_embedding import load_model, save_embeddings, save_model
import pandas as pd
import json
import time
import time
from utils import get_memory_usage, preprocess_text, save_embedding_stats
import argparse
import yaml
from tqdm import tqdm
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from scipy.sparse import vstack, csr_matrix

def train_fasttext(corpus, vector_size=None, window=8, min_count=3, epochs=20, max_vocab_size=None,save_path=None):
    """
    Train a FastText model using a text corpus.

    Args:
        corpus (list): List of text documents.
        vector_size (int): Dimensionality of the embeddings.
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Ignores words with total frequency lower than this.
        epochs (int): Number of epochs to train.
        save_path (str): Path to save the trained model.

    Returns:
        model: Trained FastText model.
    """
    print("Training FastText model with params: vector_size={}, window={}, min_count={}, epochs={} max_vocab_size={}".format(
        vector_size, window, min_count, epochs,max_vocab_size))
    start_time = time.time()
    initial_memory = get_memory_usage()

    tokenized_corpus = [text.split() for text in corpus]
    model = FastText(vector_size=vector_size, window=window,
                     max_vocab_size=max_vocab_size,
                     min_count=min_count, sg=1, workers=4)
    model.build_vocab(tokenized_corpus)
    model.train(tokenized_corpus, total_examples=len(
        tokenized_corpus), epochs=epochs)

    end_time = time.time()
    final_memory = get_memory_usage()
    training_time = end_time - start_time
    memory_used = final_memory - initial_memory

    # Collect training statistics
    training_stats = {
        "training_time_seconds": training_time,
        "memory_usage_mb": memory_used,
        "documents_per_second": len(corpus)/training_time,
        "total_documents": len(corpus),
        "model_parameters": {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs
        },
        "vocabulary_size": len(model.wv.key_to_index),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"FastText model trained successfully in {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Training speed: {len(corpus)/training_time:.2f} documents per second")

    if save_path:
        # Ensure save_path ends with .model
        if not save_path.endswith('.model'):
            save_path = save_path + '.model'
        save_model(model, save_path)
        save_embedding_stats(training_stats, save_path.replace('.model', '_model_stats.json'))
    return model

def fine_tune_fasttext(existing_model_path, corpus, vector_size=None, window=8, min_count=3, epochs=20):
    """
    Fine-tune an existing FastText model using a new text corpus.

    Args:
        existing_model_path (str): Path to the pre-trained FastText model.
        corpus (list): List of text documents to fine-tune the model on.
        vector_size (int): Dimensionality of the embeddings (optional, can match the existing model).
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Ignores words with total frequency lower than this.
        epochs (int): Number of epochs to train.

    Returns:
        model: Fine-tuned FastText model.
    """
    print(f"Fine-tuning FastText model from {existing_model_path} with params: vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")
    
    # Load the existing model
    model = FastText.load(existing_model_path)

    # Prepare the tokenized corpus
    tokenized_corpus = [text.split() for text in corpus]

    # Continue training (fine-tuning)
    model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=epochs)

    return model


def generate_embeddings(model:Union[FastText,KeyedVectors], data:list, stats_path="stats.json"):
    """
    Generate embeddings for text data in batches.

    Args:
        model: Trained FastText model
        data: DataFrame containing text data
        batch_size: Number of samples to process in each batch

    Returns:
        DataFrame with embeddings
    """
    start_time = time.time()
    initial_memory = get_memory_usage()
    total_samples = len(data)

    all_embeddings = []
    error_count = 0

    # Tokenize all texts first
    tokenized_texts = [word_tokenize(text) for text in tqdm(data, desc="Tokenizing texts")]
    
    # Process embeddings in bulk
    for tokens in tqdm(tokenized_texts, total=total_samples, desc="Generating embeddings"):
        try:
            if isinstance(model, FastText):
                embedding = model.wv.get_mean_vector(keys=tokens)
            else:
                embedding = model.get_mean_vector(keys=tokens)
            all_embeddings.append(embedding)
        except Exception as e:
            # Handle errors (e.g., empty token list or words not in vocabulary)
            print(f"Error generating embedding: {str(e)}")
            all_embeddings.append(np.zeros(model.vector_size))
            error_count += 1

    # Calculate performance metrics
    end_time = time.time()
    final_memory = get_memory_usage()

    processing_time = end_time - start_time
    memory_used = final_memory - initial_memory
    successful_samples = total_samples - error_count

    print(f"\nEmbedding Generation Statistics:")
    print(f"Total time: {processing_time:.2f} seconds")
    print(
        f"Processing speed: {successful_samples/processing_time:.2f} samples per second")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Successful samples: {successful_samples}/{total_samples}")
    print(f"Error rate: {(error_count/total_samples)*100:.2f}%")
    embedding_generation_stats = {
        "processing_time_seconds": processing_time,
        "memory_usage_mb": memory_used,
        "successful_samples": successful_samples,
        "total_samples": total_samples,
        "error_rate": error_count/total_samples,
        "samples_per_second": successful_samples/processing_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_embedding_stats(embedding_generation_stats, stats_path)
    # Convert numpy array to sparse CSR matrix for memory efficiency
    sparse_embeddings = [csr_matrix(embedding.reshape(1, -1)) for embedding in all_embeddings]
    return vstack(sparse_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FastText embeddings.")
    parser.add_argument("--input_path", type=str, required=False,
                        help="Path to the input dataset (CSV).")
    parser.add_argument("--output_path", type=str,
                        required=True, help="Path to save the embeddings.")
    parser.add_argument("--model_save_path", type=str,
                        required=True, help="Path to save the FastText model.")
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

    if args.input_path is None and (args.mode == "train" or args.mode == "training"):
        input_path = f"data/processed/{config['dataset']['train']}.csv"
    elif args.input_path is None and (args.mode == "encode" or args.mode == "encoding"):
        input_path = f"data/processed/{config['dataset']['test']}.csv"
    elif args.input_path is None and (args.mode == "val"):
        input_path = f"data/processed/{config['dataset']['val']}.csv"
    else:
        input_path = args.input_path

    store_type = config["embeddings"]["store_type"] if "store_type" in config["embeddings"] else "csv"
    if not args.model_save_path.endswith('.model'):
        args.model_save_path = args.model_save_path + '.model'

    # Load dataset
    print(f"Loading dataset from {input_path}")
    data = pd.read_csv(input_path)

    pre_process_params = config["embeddings"]['pre_process'] if "pre_process" in config["embeddings"] else None
    if pre_process_params:
        data["text"] = data["text"].apply(partial(preprocess_text, **pre_process_params)) 
        
    pre_trained= config["embeddings"]["params"]["pre_trained"] if  "pre_trained" in config["embeddings"]["params"] else False
    
    # Remove empty texts
    data = data.dropna(subset=["text"])
    data = data[data["text"] != '']

    # Train or load FastText model
    if pre_trained:
        print("Loading pre-trained model")
        model = KeyedVectors.load_word2vec_format("data/external/cc.en.300.vec")
    elif args.mode == "encode":
        model = load_model(args.model_save_path, model_type="fasttext")
    else:
        label_count = len(
            data[config["dataset"]["label_column"]].unique().tolist())
        num_samples = min(label_count*1000, len(data))
        print(f"Using {num_samples} samples for training embedding model")
        corpus = data.sample(num_samples)["text"].tolist()
        del config["embeddings"]['pre_process']
        model = train_fasttext(
            corpus, save_path=args.model_save_path, **config["embeddings"]["params"])

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(
        model, data["text"].tolist(), stats_path=stats_path)
    
    if store_type == "csv":
        # Convert sparse matrix to dense format for CSV
        data["embeddings"] = embeddings.toarray().tolist()
        save_embeddings(data.dropna(subset=["embeddings"]), args.output_path)
    elif store_type == "npz":
        labels = data["label"].to_numpy()
        save_embeddings({"X":embeddings,"y":labels}, args.output_path, store_type=store_type)

    print("Done!")
