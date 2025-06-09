from functools import partial
import time
import pandas as pd
from tqdm import tqdm
from base_embedding import save_embeddings
import json
from sentence_transformers import SentenceTransformer
import random
from utils import *

def generate_cde_embeddings(corpus,stats_path):
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
    
    model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
    minicorpus_size = model[0].config.transductive_corpus_size
    # Put some strings here that are representative of your corpus, for example by calling random.sample(corpus, k=minicorpus_size)
    minicorpus_docs = random.sample(corpus, k=minicorpus_size)
    # You must use exactly this many documents in the minicorpus. You can oversample if your corpus is smaller.
    assert len(minicorpus_docs) == minicorpus_size

    dataset_embeddings = model.encode(
        minicorpus_docs,
        prompt_name="document",
        convert_to_tensor=True
    )

    doc_embeddings = []
    for ind, doc in tqdm(enumerate(corpus), total=len(corpus)):
        # Skip empty docs
        if not doc:
            doc_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
            continue

        # Check if the document exceeds token limit
        # If it does, split into chunks and use mean pooling
        if len(doc.split()) > 512:
            # Split document into chunks of approximately 500 words
            words = doc.split()
            chunks = [' '.join(words[i:i+500]) for i in range(0, len(words), 500)]
            
            # Encode each chunk
            chunk_embeddings = []
            for chunk in chunks:
                emb = model.encode(
                    chunk,
                    prompt_name="document",
                    dataset_embeddings=dataset_embeddings,
                    convert_to_tensor=True,
                )
                chunk_embeddings.append(emb)
            
            # Compute mean embedding across all chunks
            mean_embedding = np.mean(chunk_embeddings, axis=0)
            doc_embeddings.append(mean_embedding)
            continue
        doc_embeddings.append(model.encode(
            doc,
            prompt_name="document",
            dataset_embeddings=dataset_embeddings,
            convert_to_tensor=True,
        ))
        
    # Collect training statistics
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
            "minicorpus_size": minicorpus_size,
            "model_name": "jxm/cde-small-v1",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"CDE model trained successfully in {training_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} MB")
    print(f"Training speed: {len(corpus)/training_time:.2f} documents per second")
    save_embedding_stats(training_stats, stats_path)
    return doc_embeddings


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
    
    pre_process_params = config["embeddings"]["pre_process"] if "pre_process" in config["embeddings"] else None
    if pre_process_params:
        data["text"] = data["text"].apply(partial(preprocess_text, **pre_process_params)) 
        
    corpus = data["text"].tolist()  # Replace 'text' if your text column has a different name

    # Generate embeddings
    embeddings = generate_cde_embeddings(corpus,stats_path)
    embeddings = [ x.cpu() for x in embeddings]
    data["embeddings"] = embeddings
    data["embeddings"] = data["embeddings"].apply(lambda x: json.dumps(x.tolist()))

    # Save embeddings
    save_embeddings(data, args.output_path)