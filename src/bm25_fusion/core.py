"""
BM25 Fusion package initialization.
"""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-positional-arguments

import gc
from threading import Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
from numba import njit, prange
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as st
from .tokenization import tokenize_texts

class BM25:
    """
    BM25 class for information retrieval.
    """
    def __init__(self, texts, **kwargs):
        """
        Initialize BM25 instance.
        """
        assert texts is not None, "Text for BM25 cannot be empty / None."

        self.k1 = kwargs.get('k1', 1.5)
        self.b = kwargs.get('b', 0.75)
        self.delta = kwargs.get('delta', 0.5)
        self.variant = kwargs.get('variant', 'bm25').lower()
        self.stopwords = set(s.lower() for s in kwargs.get('stopwords', [])) \
            if kwargs.get('stopwords') is not None else set(st.words('english'))
        # Tokenize texts
        corpus_tokens = tokenize_texts(texts, num_processes=kwargs.get('num_processes', 4))
        self.doc_lengths = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = np.mean(self.doc_lengths)
        self.stemmer = PorterStemmer()
        self.num_docs = len(texts)
        self.texts = texts if texts is not None else [""] * self.num_docs

        # Compute the stemmed corpus once:
        stemmed_corpus = self._stem_corpus(corpus_tokens)
        self.vocab = self._build_vocab(stemmed_corpus)
        self.tf_matrix = self._compute_tf_matrix(stemmed_corpus)
        del stemmed_corpus  # Free up memory.

        gc.collect()

        self.idf = self._compute_idf()
        self.metadata = kwargs.get('metadata', [{} for _ in range(self.num_docs)])

        # Precompute lower-case texts for efficient keyword matching.
        self.texts_lower = [t.lower() for t in self.texts]

        # Determine method code:
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
            self.tf_matrix[0], self.tf_matrix[1], self.tf_matrix[2],
            self.idf, self.doc_lengths, self.avgdl, self._method_code,
            self.k1, self.b, self.delta
        )
        # Setup lock for live updates.
        self.lock = Lock()

    def _stem_corpus(self, corpus):
        """
        Apply stemming to the corpus in parallel.
        """

        def stem_doc(doc):
            return [self.stemmer.stem(word) for word in doc]

        with ThreadPoolExecutor() as executor:
            stemmed = list(executor.map(stem_doc, corpus))
        return stemmed

    def _build_vocab(self, corpus):
        """
        Build vocabulary from the corpus in parallel.
        """

        def unique_words(doc):
            return set(doc)

        with ThreadPoolExecutor() as executor:
            sets = list(executor.map(unique_words, corpus))

        unique_words_set = set().union(*sets)
        return {word: i for i, word in enumerate(unique_words_set)}

    def _compute_tf_matrix(self, corpus):
        """
        Compute term frequency arrays using plain Python loops.
        Returns:
            tf_data: list of term frequencies (float)
            tf_indices: list of vocabulary indices (int)
            tf_indptr: list of document pointer indices (int)
        """
        data_list = []
        indices_list = []
        indptr = [0]
        for doc in corpus:
            counts = Counter(doc)
            for word, count in counts.items():
                vocab_index = self.vocab.get(word)
                if vocab_index is not None:
                    indices_list.append(vocab_index)
                    data_list.append(float(count))
            indptr.append(len(data_list))
        self.vocab_size = len(self.vocab)
        # Minimal conversion to numpy arrays for Numba interoperability.
        return (np.array(data_list, dtype=np.float32),
                np.array(indices_list, dtype=np.int32),
                np.array(indptr, dtype=np.int32))

    def _compute_idf(self):
        """
        Compute inverse document frequency.
        """
        df = np.array(self.tf_matrix[0].astype(bool).sum(axis=0)).flatten()
        df = np.maximum(df, 1e-6)
        return np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

    def query(self, query_tokens, metadata_filter=None, top_k=10, do_keyword=True):
        """
        Query the BM25 index.
        """
        query_tokens = query_tokens if isinstance(query_tokens, list) else query_tokens.split()
        assert len(query_tokens) > 0, "Query tokens cannot be empty."

        query_tokens = [
            self.stemmer.stem(token.lower())
            for token in query_tokens
            if not self.stopwords or token.lower() not in self.stopwords
        ]
        
        assert len(query_tokens) > 0, """Query tokens must include words \
            beyond the provided stop-words."""

        qvec = [0.0] * len(self.vocab)
        for word in query_tokens:
            if word in self.vocab:
                qvec[self.vocab[word]] += 1

        qvec_np = np.array(qvec, dtype=np.float32)
        scores = _retrieve_scores(self.eager_index, self.tf_matrix[1], self.tf_matrix[2], qvec_np)

        # Convert self.texts_lower and keywords to tuples to fix Numba warnings.
        if do_keyword:
            keywords = tuple(token.lower() for token in query_tokens)
            scores += _compute_keyword_scores(tuple(self.texts_lower), keywords)

        if metadata_filter:
            mask = np.array(
                [
                    1.0 if all(self.metadata[i].get(k) == v for k, v in metadata_filter.items())
                    else 0.0
                    for i in range(self.num_docs)
                ],
                dtype=np.float32,
            )
            scores *= mask

        top_indices = np.argsort(-scores)[:top_k]

        results = [
            {"text": self.texts[i], "score": float(scores[i]), **self.metadata[i]}
            for i in top_indices
            if scores[i] > 0
        ]
        return results

    def save(self, filepath):
        """
        Save the BM25 index weights and parameters using Joblib.
        """
        state = {
            'k1': self.k1,
            'b': self.b,
            'delta': self.delta,
            'variant': self.variant,
            'stopwords': list(self.stopwords),
            'num_docs': self.num_docs,
            'doc_lengths': self.doc_lengths,
            'avgdl': self.avgdl,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'tf_matrix': (
                 self.tf_matrix[0],
                 self.tf_matrix[1],
                 self.tf_matrix[2]
            ),
            'idf': self.idf,
            'metadata': self.metadata,
            'texts': self.texts,
            'texts_lower': self.texts_lower,
            '_method_code': self._method_code,
            'eager_index': self.eager_index
        }
        joblib.dump(state, filepath, compress=3)

    @staticmethod
    def load(filepath):
        """
        Load the BM25 index weights and parameters from a file using Joblib.
        """
        state = joblib.load(filepath)
        obj = BM25.__new__(BM25)  # create an uninitialized BM25 instance
        obj.__dict__.update(state)
        # Recreate any non-serializable attributes if needed (e.g. stemmer)
        obj.stemmer = PorterStemmer()
        obj.lock = Lock()
        return obj

    def _rebuild_index(self, num_processes=4):
        """
        Rebuild the BM25 index from the current texts.
        """
        tokenized_texts = tokenize_texts(self.texts, num_processes=num_processes)
        self.doc_lengths = np.array([len(doc) for doc in tokenized_texts], dtype=np.float32)
        self.avgdl = np.mean(self.doc_lengths) if self.doc_lengths.size > 0 else 0.0
        stemmed_corpus = self._stem_corpus(tokenized_texts)
        self.vocab = self._build_vocab(stemmed_corpus)
        self.tf_matrix = self._compute_tf_matrix(stemmed_corpus)
        del stemmed_corpus
        gc.collect()
        self.idf = self._compute_idf()
        self.eager_index = _eager_scores(
            self.tf_matrix[0], self.tf_matrix[1], self.tf_matrix[2],
            self.idf, self.doc_lengths, self.avgdl, self._method_code,
            self.k1, self.b, self.delta
        )

    def add_document(self, new_text:list, new_metadata:list=None, num_processes=4):
        """
        Add a new document to the index based on its text.
        new_text: a single document string.
        new_metadata: a metadata dict for the document.
        """
        with self.lock:
            self.texts.extend(new_text)
            self.texts_lower.extend([n.lower() for n in new_text])
            self.metadata.extend(new_metadata if new_metadata is not None else [{}])
            self.num_docs += len(new_text)
            # Rebuild the index with the new document incorporated.
            self._rebuild_index(num_processes=num_processes)

    def remove_document(self, text):
        """
        Remove the first document matching the provided text.
        """
        with self.lock:
            try:
                idx = self.texts.index(text)
            except ValueError as e:
                raise ValueError("Document matching the provided text not found.") from e
            del self.texts[idx]
            del self.texts_lower[idx]
            del self.metadata[idx]
            self.num_docs -= 1
            # Rebuild the index after removal.
            self._rebuild_index()

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
            if texts[i].find(keyword) != -1:
                keyword_scores[i] += 1
    return keyword_scores

