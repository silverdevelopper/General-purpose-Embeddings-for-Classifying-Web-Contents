# src/embeddings/doc2vec_embedding.py

import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from base_embedding import save_embeddings, save_model
import pandas as pd
import json
from functools import partial
import yaml
from utils import get_memory_usage,save_embedding_stats, preprocess_text
import numpy as np
from scipy.sparse import csr_matrix
import argparse


def prepare_tagged_documents(corpus):
    """
    Convert text corpus into TaggedDocument format for Doc2Vec training.

    Args:
        corpus (list): List of text documents.

    Returns:
        list: List of TaggedDocument objects.
    """
    return [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(corpus)]


def train_doc2vec(corpus, vector_size=1000, window=10, min_count=3, epochs=20, save_path=None,dm=0):
    print("Training Doc2Vec model with params: vector_size={}, window={}, min_count={}, epochs={}".format(vector_size, window, min_count, epochs))
    """
    Train a Doc2Vec model using a text corpus.

    Args:
        corpus (list): List of text documents.
        vector_size (int): Dimensionality of the embeddings.
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Ignores words with total frequency lower than this.
        epochs (int): Number of epochs to train.
        save_path (str): Path to save the trained model.

    Returns:
        model: Trained Doc2Vec model.
    """
    start_time = time.time()
    initial_memory = get_memory_usage()
    tagged_documents = prepare_tagged_documents(corpus)
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=8,dm=dm)
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
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
        save_model(model, save_path)
        save_embedding_stats(training_stats, save_path.replace('.model', '_train_stats.json'))

    return model

def generate_embeddings(data:pd.DataFrame,model:Doc2Vec,stats_path,store_type="csv"):
    pbar = tqdm(total=len(data), desc="Generating embeddings")
    error_count = 0
    
    start_time = time.time()
    initial_memory = get_memory_usage()
    total_samples = len(data)
    
    data["embeddings"] = None
    for idx, row in data.iterrows():
        try:
            if store_type == "csv":
                data.at[idx, "embeddings"] =  json.dumps(model.infer_vector(row["text"].split()).tolist()) #infer_vectors(model, row["text"]).tolist()
            elif store_type == "npz":
                data.at[idx, "embeddings"] = model.infer_vector(row["text"].split())
        except Exception as e:
            print(f"Error encoding row {idx}")
            error_count += 1
        pbar.update(1)
        
    pbar.close()
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
    if store_type == "csv":
        return data.dropna(subset=["embeddings"])
    elif store_type == "npz":
        # Convert embeddings to a CSR matrix efficiently
        embeddings_list = np.vstack(data["embeddings"].values)
        embeddings_matrix = csr_matrix(embeddings_list)
        return {"X":embeddings_matrix,"y":data["label"].to_numpy()}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Doc2Vec embeddings.")
    parser.add_argument("--input_path", type=str, required=False,default=None, help="Path to the input dataset (CSV).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the embeddings.")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the Doc2Vec model.")
    parser.add_argument("--config", type=str, required=True, help="Path to save the Doc2Vec model.")
    parser.add_argument("--mode", default="train", type=str, required=True, help="Mode of operation: train or encode.")
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
    pre_process_params = config["embeddings"]['pre_process'] if "pre_process" in config["embeddings"] else None
    store_type = config["embeddings"]["store_type"] if "store_type" in config["embeddings"] else "csv"
    if pre_process_params:
        print(f"Preprocessing text with parameters: {pre_process_params}")
        data["text"] = data["text"].apply(partial(preprocess_text, **pre_process_params)) 
    
    corpus = data["text"].tolist()  # Replace 'text' if your text column has a different name

    # Train Doc2Vec model and infer embeddings
    if args.mode == "train":
        if "pre_process" in config["embeddings"] :
            del config["embeddings"]['pre_process']  # Remove pre_process from params
        model = train_doc2vec(corpus, save_path=args.model_save_path,**config["embeddings"]["params"])
    elif args.mode == "encode" or args.mode == "val":
        model = Doc2Vec.load(args.model_save_path)
    else:
        raise ValueError("Invalid mode. Use 'train' or 'encode'.")
    
    data = generate_embeddings(data, model, stats_path,store_type=store_type)

    # Save embeddings
    save_embeddings(data, args.output_path, store_type=store_type)