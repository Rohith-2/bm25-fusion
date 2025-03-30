"""
Tokenization module for BM25 Fusion.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import nltk
from tqdm import tqdm
from nltk.stem import PorterStemmer
import dask.bag as db
from dask.diagnostics import ProgressBar
import numpy as np

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Set up the tokenizer; you can use any NLTK tokenizer.
tokenizer = nltk.word_tokenize

@lru_cache(maxsize=10000)
def tokenize(text):
    """
    Tokenizes the input text using word boundaries and applies stemming.
    Uses LRU cache to avoid re-tokenizing the same text.
    """
    tokens = re.findall(r'\b\w+\b', text)
    return [stemmer.stem(token) for token in tokens]

@lru_cache(maxsize=10000)
def whitespace_tokenize(text):
    """
    Tokenizes the input text using whitespace.
    Uses LRU cache to avoid re-tokenizing the same text.
    """
    return text.split()

@lru_cache(maxsize=10000)
def punctuation_tokenize(text):
    """
    Tokenizes the input text by splitting on punctuation.
    Uses LRU cache to avoid re-tokenizing the same text.
    """
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

@lru_cache(maxsize=10000)
def process_document(doc):
    """
    Processes a document using the default tokenizer.
    Uses LRU cache to avoid re-tokenizing the same document.
    """
    return tokenizer(doc)

def tokenize_texts(texts, tokeniser=process_document, num_partitions=None):
    """
    Tokenize a list of texts in parallel using Dask.
    Automatically determines optimal number of partitions based on data size.

    :param texts: List of raw text strings.
    :param num_partitions: Number of partitions for Dask (optional).
    :return: List of tokenized documents.
    """
    if num_partitions is None:
        # Calculate optimal number of partitions based on data size
        num_partitions = min(32, max(4, len(texts) // 1000))
    
    print(f"Tokenizing documents in parallel with {num_partitions} partitions:", end=' ')
    bag_obj = db.from_sequence(texts, npartitions=num_partitions)
    
    # Use persist() to cache intermediate results
    with ProgressBar():
        corpus_tokens = bag_obj.map(tokeniser).persist()
    
    return corpus_tokens

if __name__ == "__main__":
    SAMPLE_TEXT = "This is a sample text for tokenization."
    print("Default Tokenization:", tokenize(SAMPLE_TEXT))
    print("Whitespace Tokenization:", whitespace_tokenize(SAMPLE_TEXT))
    print("Punctuation Tokenization:", punctuation_tokenize(SAMPLE_TEXT))
