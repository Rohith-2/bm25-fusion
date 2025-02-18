"""
BM25 Fusion package initialization.
"""

import pickle
from collections import defaultdict
from nltk.corpus import stopwords as st
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')

class BM25:
    """
    BM25 class for information retrieval.
    """
    def __init__(self, corpus_tokens, texts, **kwargs):
        """
        Initialize BM25 instance.
        """
        self.k1 = kwargs.get('k1', 1.5)
        self.b = kwargs.get('b', 0.75)
        self.delta = kwargs.get('delta', 0.5)
        self.variant = kwargs.get('variant', 'bm25').lower()
        self.stopwords = set(s.lower() for s in kwargs.get('stopwords', [])) \
            if kwargs.get('stopwords') is not None else set(st.words('english'))
        self.num_docs = len(corpus_tokens)
        self.doc_lengths = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = np.mean(self.doc_lengths)
        self.stemmer = PorterStemmer()
        self.vocab = self._build_vocab(self._stem_corpus(corpus_tokens))
        self.tf_matrix = self._compute_tf_matrix(self._stem_corpus(corpus_tokens))
        self.idf = self._compute_idf()
        self.metadata = kwargs.get('metadata', [{} for _ in range(self.num_docs)])
        self.texts = texts if texts is not None else [""] * self.num_docs

        # Determine method code:
        # 0: BM25 classic, 1: BM25+, 2: BM25L, 3: ATIRE (classic BM25 with nonnegative idf)
        if self.variant in ("bm25", "lucene", "robertson"):
            self._method_code = 0
        elif self.variant == "bm25+":
            self._method_code = 1
        elif self.variant == "bm25l":
            self._method_code = 2
        elif self.variant == "atire":
            self._method_code = 3
            self.idf = np.maximum(self.idf, 0)
        else:
            raise ValueError(f"Unknown BM25 variant: {self.variant}")

        self.eager_index = _eager_scores(
            self.tf_matrix.data, self.tf_matrix.indices, self.tf_matrix.indptr,
            self.idf, self.doc_lengths, self.avgdl, self._method_code,
            self.k1, self.b, self.delta
        )

    def _stem_corpus(self, corpus):
        """
        Apply stemming to the corpus.
        """
        return [[self.stemmer.stem(word) for word in doc] for doc in corpus]

    def _build_vocab(self, corpus):
        """
        Build vocabulary from the corpus.
        """
        unique_words = set()
        for doc in corpus:
            unique_words.update(doc)
        return {word: i for i, word in enumerate(unique_words)}

    def _compute_tf_matrix(self, corpus):
        """
        Compute term frequency matrix.
        """
        row, col, data = [], [], []
        for doc_id, doc in enumerate(corpus):
            counts = defaultdict(int)
            for word in doc:
                counts[word] += 1
            for word, count in counts.items():
                if word in self.vocab:
                    row.append(doc_id)
                    col.append(self.vocab[word])
                    data.append(count)
        return csr_matrix((np.array(data, dtype=np.float32),
                           (np.array(row), np.array(col))),
                          shape=(self.num_docs, len(self.vocab)), dtype=np.float32)

    def _compute_idf(self):
        """
        Compute inverse document frequency.
        """
        df = np.array(self.tf_matrix.astype(bool).sum(axis=0)).flatten()
        df = np.maximum(df, 1e-6)
        return np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

    def query(self, query_tokens, metadata_filter=None, top_k=10, do_keyword=True):
        """
        Query the BM25 index.
        """
        if type(query_tokens)!=list:
            query_tokens = query_tokens.split()

        assert len(query_tokens)>0, "Query tokens cannot be empty."
        # Remove stopwords from query tokens if provided
        if self.stopwords is not None:
            query_tokens = [token for token in query_tokens if token.lower() not in self.stopwords]
        query_tokens = [self.stemmer.stem(token) for token in query_tokens]
        qvec = np.zeros(len(self.vocab), dtype=np.float32)
        for word in query_tokens:
            if word in self.vocab:
                qvec[self.vocab[word]] += 1
        scores = _retrieve_scores(self.eager_index, self.tf_matrix.indices,
                                  self.tf_matrix.indptr, qvec)
        
        # Calculate keyword match scores using query tokens
        if do_keyword:
            keyword_scores = _compute_keyword_scores(self.texts, query_tokens)
            scores += keyword_scores

        if metadata_filter:
            mask = np.array([
                all(self.metadata[i].get(key) == val for key, val in metadata_filter.items())
                for i in range(self.num_docs)
            ], dtype=np.float32)
            scores *= mask

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                res = {"text": self.texts[i], "score": float(scores[i])}
                res.update(self.metadata[i])
                results.append(res)
        return results

    def save(self, path):
        """
        Save the BM25 index to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load the BM25 index from a file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

@njit(parallel=True)
def _eager_scores(tf_data, tf_indices, tf_indptr, idf, doc_lengths,
                  avgdl, method_code, k1, b, delta):
    """
    Compute the eager scores for the BM25 index.
    """
    num_docs = len(doc_lengths)
    score_data = np.empty_like(tf_data)
    for d in prange(num_docs):
        norm = k1 * (1 - b + b * doc_lengths[d] / avgdl)
        for j in range(tf_indptr[d], tf_indptr[d+1]):
            tf = tf_data[j]
            if method_code in {0, 3}:
                score = idf[tf_indices[j]] * ((tf * (k1 + 1)) / (tf + norm))
            elif method_code == 1:
                score = idf[tf_indices[j]] * (((tf + delta) * (k1 + 1)) / (tf + norm + delta))
            elif method_code == 2:
                score = idf[tf_indices[j]] * (tf / (tf + norm + delta * (doc_lengths[d] / avgdl)))
            else:
                score = 0.0
            score_data[j] = score
    return score_data

@njit(parallel=True)
def _retrieve_scores(eager_data, tf_indices, tf_indptr, query_vec):
    """
    Retrieve the scores for the query.
    """
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
    """
    Compute keyword match scores for each document.
    """
    num_docs = len(texts)
    keyword_scores = np.zeros(num_docs, dtype=np.float32)
    for i in prange(num_docs):
        for keyword in keywords:
            if keyword.lower() in texts[i].lower():
                keyword_scores[i] += 1
    return keyword_scores
