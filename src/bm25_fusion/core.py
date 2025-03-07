"""
BM25 Fusion package initialization.
"""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-positional-arguments

import gc
import pickle
from threading import Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import h5py
import joblib
import numpy as np
from tqdm import tqdm
from numba import njit, prange
from numba.typed import List as TypedList
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
        self.stemmer = PorterStemmer() if kwargs.get('stemmer') is None else kwargs.get('stemmer')
        self.num_docs = len(texts)
        self.texts = texts if texts is not None else [""] * self.num_docs
        self.do_stem = kwargs.get('do_stem', False)

        # Compute the stemmed corpus once:
        if self.do_stem:
            corpus_tokens = self._stem_corpus(corpus_tokens)

        self.vocab = self._build_vocab(corpus_tokens)
        self.tf_matrix = self._compute_tf_matrix(corpus_tokens)
        del corpus_tokens  # Free up memory.

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
            stemmed = list(executor.map(stem_doc, tqdm(corpus)))
        return stemmed

    def _build_vocab(self, corpus):
        """
        Build vocabulary from the corpus in parallel.
        """

        def unique_words(doc):
            return set(doc)

        with ThreadPoolExecutor() as executor:
            sets = list(executor.map(unique_words, tqdm(corpus)))

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
        for doc in tqdm(corpus):
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
        query_tokens = query_tokens if isinstance(query_tokens, list) else query_tokens.split(" ")
        assert len(query_tokens) > 0 or metadata_filter,\
                "Query tokens or metadata cannot be empty"

        query_tokens = [
            self.stemmer.stem(token.lower())
            for token in query_tokens
            if not self.stopwords or token.lower() not in self.stopwords
        ]

        assert len(query_tokens) > 0 or metadata_filter,\
              "Query tokens must include words beyond the provided stop-words."

        qvec = [0.0] * len(self.vocab)
        for word in query_tokens:
            if word in self.vocab:
                qvec[self.vocab[word]] += 1

        qvec_np = np.array(qvec, dtype=np.float32)
        scores = _retrieve_scores(self.eager_index, self.tf_matrix[1], self.tf_matrix[2], qvec_np)

        if do_keyword:
            # Convert self.texts_lower and keywords to numba.typed.List instead of tuples.
            t_texts_lower = TypedList(self.texts_lower)
            t_keywords = TypedList([token.lower() for token in query_tokens])
            scores += _compute_keyword_scores(t_texts_lower, t_keywords)

        if metadata_filter:
            mask = np.array(
                [
                    sum(self.metadata[i].get(k) in v for k, v in metadata_filter.items())
                    if any(self.metadata[i].get(k) in v for k, v in metadata_filter.items())
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
        Save the BM25 index state using joblib with gzip compression.
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
        joblib.dump(state, filepath, compress=('gzip', 3))

    @staticmethod
    def load(filepath):
        """
        Load the BM25 index state using joblib with gzip decompression.
        """
        state = joblib.load(filepath)
        obj = BM25.__new__(BM25)  # Create an uninitialized BM25 instance.
        obj.__dict__.update(state)
        # Recreate non-serializable attributes.

        obj.stemmer = PorterStemmer()
        obj.lock = Lock()
        return obj

    def save_hdf5(self, filepath):
        """
        Save the BM25 index using HDF5.
        Numeric data are stored as datasets, and non-numeric objects are
        pickled and stored as byte arrays.
        """
        with h5py.File(filepath, "w") as f:
            # Save scalar parameters as attributes.
            f.attrs["k1"] = self.k1
            f.attrs["b"] = self.b
            f.attrs["delta"] = self.delta
            f.attrs["variant"] = self.variant
            f.attrs["num_docs"] = self.num_docs
            f.attrs["_method_code"] = self._method_code

            # Save numeric arrays.
            f.create_dataset("doc_lengths", data=self.doc_lengths)
            f.create_dataset("avgdl", data=np.array([self.avgdl]))
            f.create_dataset("idf", data=self.idf)
            # tf_matrix is a tuple of arrays.
            f.create_dataset("tf_matrix_0", data=self.tf_matrix[0])
            f.create_dataset("tf_matrix_1", data=self.tf_matrix[1])
            f.create_dataset("tf_matrix_2", data=self.tf_matrix[2])
            # Save eager_index as a dataset.
            f.create_dataset("eager_index", data=self.eager_index)

            # For Python objects (vocab, metadata, texts, texts_lower), pickle them.
            f.create_dataset("vocab", \
                             data=np.void(
                                 pickle.dumps(self.vocab, protocol=pickle.HIGHEST_PROTOCOL)
                                 ))
            f.create_dataset("metadata", \
                             data=np.void(
                                 pickle.dumps(self.metadata, protocol=pickle.HIGHEST_PROTOCOL)
                                 ))
            f.create_dataset("texts", \
                             data=np.void(
                                 pickle.dumps(self.texts, protocol=pickle.HIGHEST_PROTOCOL)
                                 ))
            f.create_dataset("texts_lower", \
                             data=np.void(
                                 pickle.dumps(self.texts_lower, protocol=pickle.HIGHEST_PROTOCOL)
                                 ))
            # Stopwords can be stored as a numpy string array.
            stopwords_list = list(self.stopwords)
            f.create_dataset("stopwords", data=np.array(stopwords_list, dtype="S"))

    @staticmethod
    def load_hdf5(filepath):
        """
        Load the BM25 index from an HDF5 file.
        The pickled objects are restored from the byte streams.
        """
        with h5py.File(filepath, "r") as f:
            obj = BM25.__new__(BM25)  # Create an uninitialized BM25 instance.

            # Load scalar parameters.
            obj.k1 = f.attrs["k1"]
            obj.b = f.attrs["b"]
            obj.delta = f.attrs["delta"]
            obj.variant = f.attrs["variant"]
            obj.num_docs = int(f.attrs["num_docs"])
            obj._method_code = int(f.attrs["_method_code"])

            # Load numeric arrays.
            obj.doc_lengths = f["doc_lengths"][:]
            obj.avgdl = float(f["avgdl"][0])
            obj.idf = f["idf"][:]
            tf0 = f["tf_matrix_0"][:]
            tf1 = f["tf_matrix_1"][:]
            tf2 = f["tf_matrix_2"][:]
            obj.tf_matrix = (tf0, tf1, tf2)
            obj.eager_index = f["eager_index"][:]

            # Restore Python objects from pickled data.
            obj.vocab = pickle.loads(bytes(f["vocab"][()]))
            obj.metadata = pickle.loads(bytes(f["metadata"][()]))
            obj.texts = pickle.loads(bytes(f["texts"][()]))
            obj.texts_lower = pickle.loads(bytes(f["texts_lower"][()]))

            # Restore stopwords (convert from bytes to string).
            obj.stopwords = set(s.decode('utf-8') for s in f["stopwords"][:])

            # Recreate any non-serializable attributes.
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
            if int(texts[i].find(keyword)) != -1:
                keyword_scores[i] += 1
    return keyword_scores
