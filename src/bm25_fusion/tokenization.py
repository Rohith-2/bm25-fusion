"""
Tokenization module for BM25 Fusion.
"""

import re
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
stemmer = PorterStemmer()

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

if __name__ == "__main__":
    SAMPLE_TEXT = "This is a sample text for tokenization."
    print("Default Tokenization:", tokenize(SAMPLE_TEXT))
    print("Whitespace Tokenization:", whitespace_tokenize(SAMPLE_TEXT))
    print("Punctuation Tokenization:", punctuation_tokenize(SAMPLE_TEXT))
