"""
Tokenization module for BM25 Fusion.
"""

import re
from concurrent.futures import ProcessPoolExecutor

import nltk
from tqdm import tqdm
from nltk.stem import PorterStemmer
import dask.bag as db
from dask.diagnostics import ProgressBar

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Set up the tokenizer; you can use any NLTK tokenizer.
tokenizer = nltk.word_tokenize

def tokenize(text):
    """
    Tokenizes the input text using word boundaries and applies stemming.
    """
    tokens = re.findall(r'\b\w+\b', text)
    return [stemmer.stem(token) for token in tokens]

def whitespace_tokenize(text):
    """
    Tokenizes the input text using whitespace.
    """
    return text.split()

def punctuation_tokenize(text):
    """
    Tokenizes the input text by splitting on punctuation.
    """
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def process_document(doc):
    """
    Processes a document using the default tokenizer.
    """
    return tokenizer(doc)

def tokenize_texts(texts, tokeniser=process_document, num_partitions=8):
    """
    Tokenize a list of texts in parallel using Dask.

    :param texts: List of raw text strings.
    :param num_partitions: Number of partitions for Dask.
    :return: List of tokenized documents.
    """
    print("Tokenizing documents in parallel :",end=' ')
    bag_obj = db.from_sequence(texts, npartitions=num_partitions)
    with ProgressBar():
        corpus_tokens = bag_obj.map(tokeniser)
    return corpus_tokens

if __name__ == "__main__":
    SAMPLE_TEXT = "This is a sample text for tokenization."
    print("Default Tokenization:", tokenize(SAMPLE_TEXT))
    print("Whitespace Tokenization:", whitespace_tokenize(SAMPLE_TEXT))
    print("Punctuation Tokenization:", punctuation_tokenize(SAMPLE_TEXT))
