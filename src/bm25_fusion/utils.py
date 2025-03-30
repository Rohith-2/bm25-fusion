import numpy as np
from tqdm import tqdm
from numba import njit, prange, cuda
from numba.typed import List as TypedList
import os
import psutil
from typing import List, Dict, Optional

@njit(parallel=True, fastmath=True)
def _retrieve_scores(eager_data: np.ndarray, tf_indices: np.ndarray, 
                    tf_indptr: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """
    Optimized score retrieval using Numba with parallel processing and fastmath.
    """
    num_docs = len(tf_indptr) - 1
    scores = np.zeros(num_docs, dtype=np.float32)
    
    # Use prange for parallel processing
    for d in prange(num_docs):
        s = 0.0
        # Optimize inner loop with direct indexing
        for j in range(tf_indptr[d], tf_indptr[d+1]):
            i = tf_indices[j]
            if query_vec[i] > 0:
                s += eager_data[j]
        scores[d] = s
    return scores

@njit(parallel=True, fastmath=True)
def _compute_keyword_scores(texts: TypedList, keywords: TypedList) -> np.ndarray:
    """
    Optimized keyword score computation using Numba with parallel processing.
    """
    num_docs = len(texts)
    keyword_scores = np.zeros(num_docs, dtype=np.float32)
    
    # Use prange for parallel processing
    for i in prange(num_docs):
        text = texts[i]
        # Optimize keyword matching with direct string operations
        for keyword in keywords:
            if text.find(keyword) != -1:
                keyword_scores[i] += 1
    return keyword_scores

@njit(parallel=True, fastmath=True)
def _eager_scores(tf_data: np.ndarray, tf_indices: np.ndarray, tf_indptr: np.ndarray,
                 idf: np.ndarray, doc_lengths: np.ndarray, avgdl: float,
                 method_code: int, k1: float, b: float, delta: float) -> np.ndarray:
    """
    Optimized eager score computation using Numba with parallel processing.
    """
    num_docs = len(doc_lengths)
    score_data = np.empty_like(tf_data)
    
    # Precompute common values
    k1_plus_1 = k1 + 1
    
    # Use prange for parallel processing
    for d in prange(num_docs):
        norm = k1 * (1 - b + b * doc_lengths[d] / avgdl)
        # Optimize inner loop with direct indexing
        for j in range(tf_indptr[d], tf_indptr[d+1]):
            tf = tf_data[j]
            idx = tf_indices[j]
            idf_val = idf[idx]
            
            if method_code in (0, 3):  # Classic BM25 or ATIRE
                score = idf_val * ((tf * k1_plus_1) / (tf + norm))
            elif method_code == 1:  # BM25+
                score = idf_val * (((tf + delta) * k1_plus_1) / (tf + norm + delta))
            elif method_code == 2:  # BM25L
                score = idf_val * (tf / (tf + norm + delta * (doc_lengths[d] / avgdl)))
            else:
                score = 0.0
                
            score_data[j] = score
    return score_data

def _current_memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def _optimize_array_dtype(arr: np.ndarray) -> np.ndarray:
    """
    Optimize array data type based on its values.
    """
    if arr.dtype == np.float64:
        if np.all(np.abs(arr) <= np.finfo(np.float32).max):
            return arr.astype(np.float32)
    elif arr.dtype == np.int64:
        if np.all(np.abs(arr) <= np.iinfo(np.int32).max):
            return arr.astype(np.int32)
    return arr