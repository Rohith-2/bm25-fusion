import numpy as np
from tqdm import tqdm
from numba import njit, prange
from numba.typed import List as TypedList
import os
import psutil

@njit(parallel=True)
def _retrieve_scores(eager_data, tf_indices, tf_indptr, query_vec):
    num_docs = len(tf_indptr) - 1
    scores = np.zeros(num_docs, dtype=np.float32)
    for d in prange(num_docs):
        s = 0.0
        for j in range(tf_indptr[d], tf_indptr[d+1]):
            i = tf_indices[j]
            if query_vec[i] > 0:
                s += eager_data[j]
        scores[d] = s
    return scores

@njit(parallel=True)
def _compute_keyword_scores(texts, keywords):
    num_docs = len(texts)
    keyword_scores = np.zeros(num_docs, dtype=np.float32)
    for i in prange(num_docs):
        for keyword in keywords:
            if int(texts[i].find(keyword)) != -1:
                keyword_scores[i] += 1
    return keyword_scores

def _current_memory_usage_mb(format='beautify'):
    """
    Returns the current memory usage of the process in MB.
    """
    mem =  (psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    if format == 'beautify':
        return f"{mem:.2f} MB"
    return mem

@njit(parallel=True)
def _eager_scores(tf_data, tf_indices, tf_indptr, idf, doc_lengths,
                  avgdl, method_code, k1, b, delta):
    num_docs = len(doc_lengths)
    score_data = np.empty_like(tf_data)
    for d in prange(num_docs):
        norm = k1 * (1 - b + b * doc_lengths[d] / avgdl)
        for j in range(tf_indptr[d], tf_indptr[d+1]):
            tf = tf_data[j]
            if method_code in (0, 3):
                score = idf[tf_indices[j]] * ((tf * (k1 + 1)) / (tf + norm))
            elif method_code == 1:
                score = idf[tf_indices[j]] * (((tf + delta) * (k1 + 1)) / (tf + norm + delta))
            elif method_code == 2:
                score = idf[tf_indices[j]] * (tf / (tf + norm + delta * (doc_lengths[d] / avgdl)))
            else:
                score = 0.0
            score_data[j] = score
    return score_data