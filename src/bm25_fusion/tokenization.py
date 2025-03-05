"""
Tokenization module for BM25 Fusion.
"""

import re
from concurrent.futures import ProcessPoolExecutor

import nltk
from tqdm import tqdm
from nltk.stem import PorterStemmer

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

def tokenize_texts(texts, num_processes=8):
    """
    Tokenize a list of texts in parallel.

    :param texts: List of raw text strings.
    :param num_processes: Number of parallel processes to use.
    :return: List of tokenized documents.
    """
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        corpus_tokens = list(executor.map(process_document, tqdm(texts, total=len(texts))))
    return corpus_tokens

if __name__ == "__main__":
    SAMPLE_TEXT = "This is a sample text for tokenization."
    print("Default Tokenization:", tokenize(SAMPLE_TEXT))
    print("Whitespace Tokenization:", whitespace_tokenize(SAMPLE_TEXT))
    print("Punctuation Tokenization:", punctuation_tokenize(SAMPLE_TEXT))
