# src/embeddings/hf_embedding.py

from functools import partial
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from tqdm import tqdm
from base_embedding import save_embeddings
import json
from utils import *

def generate_hf_embeddings(corpus, tokenizer_model="bert-base-uncased",model_name="bert-base-uncased",stats_path="data/processed/hf_stats.json"):
    """
    Generate HuggingFace model-based embeddings for a corpus.

    Args:
        corpus (list): List of text documents.
        model_name (str): HuggingFace transformer model name.

    Returns:
        list: List of dense embeddings.
    """
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for text in tqdm(corpus, desc="Generating embeddings"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy())
            
    end_time = time.time()
    final_memory = get_memory_usage()
    training_time = end_time - start_time
    memory_used = final_memory - initial_memory

    training_stats = {
        "training_time_seconds": training_time,
        "memory_usage_mb": memory_used,
        "documents_per_second": len(corpus)/training_time,
        "total_documents": len(corpus),
        "model_parameters": {
            "model_name": model_name,
            "tokenizer": tokenizer_model
        },
        "vocabulary_size": len(model.wv.key_to_index),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"FastText model trained successfully in {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Training speed: {len(corpus)/training_time:.2f} documents per second")
    save_embedding_stats(training_stats, stats_path)
    return embeddings


if __name__ == "__main__":
    import argparse,yaml

    parser = argparse.ArgumentParser(description="Generate HuggingFace embeddings.")
    parser.add_argument("--input_path", type=str, required=False, help="Path to the input dataset (CSV).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the embeddings.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
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

    # Load dataset
    data = pd.read_csv(input_path)
    # Parse the ngram_range argument
    
    pre_process_params = config["embeddings"]["pre_process"] if "pre_process" in config["embeddings"] else None
    if pre_process_params:
        data["text"] = data["text"].apply(partial(preprocess_text, **pre_process_params)) 
        
    corpus = data["text"].tolist()  # Replace 'text' if your text column has a different name

    # Generate embeddings
    embeddings = generate_hf_embeddings(corpus, **config["embeddings"]["params"], stats_path=stats_path)
    data["embeddings"] = embeddings
    data["embeddings"] = data["embeddings"].apply(lambda x: json.dumps(x.tolist()))

    # Save embeddings
    save_embeddings(data, args.output_path)