import numpy as np
import json
import re
import string
import time
import pandas as pd
import psutil
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

 # pre_process:
def preprocess_text(
    text, 
    lower=True, 
    remove_punctuation=True, 
    remove_stopwords=True, 
    remove_numbers=True, 
    remove_special_characters=True, 
    remove_extra_whitespace=True, 
    lemma=False, 
    stem=False
):
    if not isinstance(text, str):
        return str(text)
    
    if lower:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if remove_special_characters:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()
        
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    if lemma:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def save_embedding_stats(stats, save_path):
    """
    Save embedding statistics to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Statistics saved to {save_path}")
    


def chunk_dataframe_rows(df, chunk_size=1000):
    """
    Split a DataFrame into chunks of given size.
    
    Args:
        df: pandas DataFrame
        chunk_size: int, number of rows in each chunk
        
    Returns:
        List of DataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size, :]  # All columns
    

def chunk_array_rows(arr, chunk_size=1000):
    """
    Split a 2D numpy array into chunks of given size.
    
    Args:
        array: 2D numpy array
        chunk_size: int, number of rows in each chunk
    Returns:
        List of array chunks
    """

    """Split array row-wise without breaking columns"""
    print(f"Chunking array of shape {arr.shape} into chunks of size {chunk_size}")
    for i in range(0, arr.shape[0], chunk_size):
        yield arr[i:i + chunk_size, :]  # Keep all columns