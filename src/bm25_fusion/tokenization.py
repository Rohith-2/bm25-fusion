"""
Tokenization module for BM25 Fusion.
Provides optimized document tokenization, word boundary extraction, and stemming routines.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import nltk
from nltk.stem import PorterStemmer
from tqdm import tqdm

# Initialize module-level PorterStemmer instance
stemmer = PorterStemmer()


def _safe_nltk_tokenize(text: str) -> List[str]:
    """
    Safely tokenize text using NLTK word_tokenize with fallback to regex word boundaries.
    """
    try:
        return nltk.word_tokenize(text)
    except Exception:
        # Fallback to regex word boundary tokenizer if NLTK data is missing
        return re.findall(r"\b\w+\b", text)


def tokenize(text: str) -> List[str]:
    """
    Tokenizes input text using word boundaries and applies Porter stemming.

    Args:
        text: Input raw text string.

    Returns:
        List of stemmed token strings.
    """
    tokens = re.findall(r"\b\w+\b", text)
    return [stemmer.stem(token) for token in tokens]


def whitespace_tokenize(text: str) -> List[str]:
    """
    Tokenizes input text using whitespace splitting.

    Args:
        text: Input raw text string.

    Returns:
        List of whitespace-separated tokens.
    """
    return text.split()


def punctuation_tokenize(text: str) -> List[str]:
    """
    Tokenizes input text by splitting on punctuation boundaries.

    Args:
        text: Input raw text string.

    Returns:
        List of word and punctuation tokens.
    """
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def process_document(doc: str) -> List[str]:
    """
    Processes a single document using NLTK or fallback tokenization.

    Args:
        doc: Document text string.

    Returns:
        List of token strings.
    """
    return _safe_nltk_tokenize(doc)


def tokenize_texts(
    texts: List[str], num_processes: Optional[int] = 4, show_progress: bool = True
) -> List[List[str]]:
    """
    Tokenizes a corpus of texts with process-pool parallelism for large corpora
    and fast in-process sequential processing for small corpora (< 200 documents).

    Args:
        texts: List of raw document text strings.
        num_processes: Maximum number of worker processes to use.
        show_progress: Whether to display a progress bar.

    Returns:
        List of tokenized documents, where each document is a list of token strings.
    """
    if not texts:
        return []

    # Small-corpus optimization: avoid IPC overhead when document count is small
    if len(texts) < 200 or (num_processes is not None and num_processes <= 1):
        iterator = tqdm(texts, total=len(texts), disable=not show_progress)
        return [_safe_nltk_tokenize(doc) for doc in iterator]

    workers = num_processes if num_processes is not None else 4
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = tqdm(texts, total=len(texts), disable=not show_progress)
        corpus_tokens = list(executor.map(process_document, iterator))

    return corpus_tokens


if __name__ == "__main__":
    SAMPLE_TEXT = "This is a sample text for tokenization."
    print("Default Tokenization:", tokenize(SAMPLE_TEXT))
    print("Whitespace Tokenization:", whitespace_tokenize(SAMPLE_TEXT))
    print("Punctuation Tokenization:", punctuation_tokenize(SAMPLE_TEXT))
