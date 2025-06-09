# src/embeddings/base_embedding.py

import os
from typing import Union
from scipy.sparse import csr_matrix
import pandas as pd
from gensim.models import KeyedVectors, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import json
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix
import numpy as np


def save_embeddings(df: Union[pd.DataFrame, csr_matrix], save_path, mode='w', store_type='csv'):
    """
    Save embeddings as a CSV file.

    Args:
        embeddings (list): List of dense embeddings or feature vectors.
        save_path (str): File path to save the embeddings as a CSV file.
    """
    if store_type == 'csv':
        if mode == 'w':
            print("Saving embeddings..")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Embeddings saved at {save_path}")
        else:
            df.to_csv(save_path, mode='a', header=False, index=False)
    elif store_type == 'parquet':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"Embeddings saved at {save_path}")
    elif store_type == 'npz':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        X = df["X"]
        y = df["y"]
        np.savez(save_path.replace(".csv", ".npz"),
                 data=X.data,
                 indices=X.indices,
                 indptr=X.indptr,
                 shape=X.shape,
                 labels=y)

        print(f"Embeddings saved at {save_path}")


def load_embeddings(load_path, label_column="category", store_type='csv'):
    """
    Load embeddings from a CSV file.

    Args:
        load_path (str): File path to load the embeddings from.

    Returns:
        list: List of dense embeddings or feature vectors.
    """
    if store_type == 'csv':
        df = pd.read_csv(load_path, converters={
                         "embeddings": lambda x: np.array(json.loads(x))})
        print("Load path:", load_path)
        embeddings = np.stack(np.array(df["embeddings"]))
        if "category_encoded" in df.columns:
            categories = df['category_encoded']
        else:
            categories = df[label_column]
    elif store_type == "npz":
        print("Loading embeddings from npz file...")
        data = np.load(load_path.replace(".csv", ".npz"))
        embeddings = csr_matrix((data['data'], 
                                data['indices'],
                                data['indptr']),
                                shape=data['shape'])
        categories = data['labels']
    return embeddings, categories


def save_model(model, save_path):
    """
    Save a training model (e.g., Doc2Vec, FastText) to a file.

    Args:
        model: The trained embedding model.
        save_path (str): Path to the model file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Handle saving for Gensim models
    if isinstance(model, KeyedVectors):  # For FastText word vectors
        model.save(save_path)
    elif hasattr(model, "save"):  # For models with a .save() method
        model.save(save_path)
    # For other models (e.g., sklearn models)
    elif hasattr(model, "dump") or isinstance(model, TfidfVectorizer):
        joblib.dump(model, save_path)
    else:
        raise ValueError("Unsupported model type for saving.")
    print(f"Model saved to {save_path}")


def load_model(load_path, model_type: str = None):
    """
    Load a saved training model (e.g., Doc2Vec, FastText) from a file.

    Args:
        load_path (str): Path to the model file.

    Returns:
        model: The loaded embedding model.
    """
    # Handle loading for Gensim models
    if model_type == "fasttext":
        return FastText.load(load_path)

    if load_path.endswith(".kv"):  # For FastText word vectors
        model = KeyedVectors.load(load_path)
    elif os.path.exists(load_path):  # For other models
        model = joblib.load(load_path)
    else:
        raise FileNotFoundError(f"Model file not found: {load_path}")
    return model
