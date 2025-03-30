"""
BM25 Fusion package initialization.
"""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-positional-arguments

import gc
import pickle
from threading import Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Optional, Union
import warnings

import h5py
import joblib
from tqdm import tqdm
from numba import njit, prange
from numba.typed import List as TypedList
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as st
from scipy.sparse import csr_matrix, save_npz, load_npz
import dask.bag as db
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar

from .tokenization import tokenize_texts
from .utils import _retrieve_scores, _compute_keyword_scores, _current_memory_usage_mb, _eager_scores

class BM25:
    """
    BM25 class for information retrieval.
    """
    def __init__(self, texts: List[str], **kwargs):
        """
        Initialize BM25 instance.
        """
        assert texts is not None, "Text for BM25 cannot be empty / None."

        self.verbose = kwargs.get('verbose', False)
        self.k1 = kwargs.get('k1', 1.5)
        self.b = kwargs.get('b', 0.75)
        self.delta = kwargs.get('delta', 0.5)
        self.variant = kwargs.get('variant', 'bm25').lower()
        self.stopwords = set(s.lower() for s in kwargs.get('stopwords', [])) \
            if kwargs.get('stopwords') is not None else set(st.words('english'))
        
        # Optimize memory usage by using smaller data types
        self.num_docs = len(texts)
        self.texts = texts if texts is not None else [""] * self.num_docs
        
        # Tokenize texts as lazy Dask bag with optimized partitioning
        num_partitions = min(32, max(4, self.num_docs // 1000))
        corpus_tokens = tokenize_texts(texts, num_partitions=num_partitions)
        
        if self.verbose:
            print(f"Tokenization complete. Memory usage: {_current_memory_usage_mb()}")

        # Compute document lengths lazily with optimized memory usage
        lengths_bag = corpus_tokens.map(len)
        self.doc_lengths = np.array(lengths_bag.compute(), dtype=np.int32)  # Use int32 instead of float32
        self.avgdl = np.mean(self.doc_lengths)
        
        self.stemmer = PorterStemmer() if kwargs.get('stemmer') is None else kwargs.get('stemmer')
        self.do_stem = kwargs.get('do_stem', False)

        # Optionally apply stemming in a lazy way with memory optimization
        if self.do_stem:
            corpus_tokens = self._stem_corpus(corpus_tokens)

        # Build vocabulary with memory optimization
        self.vocab = self._build_vocab(corpus_tokens)
        
        # Compute TF matrix with optimized memory usage
        self.tf_matrix = self._compute_tf_matrix(corpus_tokens)
        if self.verbose:
            print(f"TF Matrix Processed. Memory usage: {_current_memory_usage_mb()}")
        
        del corpus_tokens  # Free up memory
        gc.collect()

        # Compute IDF with optimized memory usage
        self.idf = self._compute_idf()
        if self.verbose:
            print(f"IDF complete. Memory usage: {_current_memory_usage_mb()}")
        
        self.metadata = kwargs.get('metadata', [{} for _ in range(self.num_docs)])

        # Precompute lower-case texts for efficient keyword matching
        self.texts_lower = [t.lower() for t in self.texts]

        # Determine method code and compute eager index
        self._method_code = self._get_method_code()
        if self._method_code == 3:  # atire variant
            self.idf = np.maximum(self.idf, 0)

        self.eager_index = _eager_scores(
            self.tf_matrix.data, self.tf_matrix.indices, self.tf_matrix.indptr,
            self.idf, self.doc_lengths, self.avgdl, self._method_code,
            self.k1, self.b, self.delta
        )
        if self.verbose:
            print(f"Eager Indexing complete. Memory usage: {_current_memory_usage_mb()}")
        
        self.lock = Lock()

    def _get_method_code(self) -> int:
        """Helper method to determine BM25 variant code."""
        if self.variant in ("bm25", "lucene", "robertson"):
            return 0
        elif self.variant == "bm25+":
            return 1
        elif self.variant == "bm25l":
            return 2
        elif self.variant == "atire":
            return 3
        else:
            raise ValueError(f"Unknown BM25 variant: {self.variant}")

    def _stem_corpus(self, corpus):
        """
        Lazily stem documents using a Dask bag with memory optimization.
        """
        def stem_doc(doc):
            return [self.stemmer.stem(word) for word in doc]
        stemmed = corpus.map(stem_doc)
        if self.verbose:
            with ProgressBar():
                stemmed = stemmed.persist()
        return stemmed

    def _build_vocab(self, corpus):
        """
        Build vocabulary lazily from the corpus with memory optimization.
        """
        def unique_words(doc):
            return set(doc)
        print("Building vocabulary from corpus")
        bag = corpus.map(unique_words)
        union_set = bag.fold(lambda a, b: a.union(b), initial=set())
        return {word: i for i, word in enumerate(union_set.compute())}

    def _compute_tf_matrix(self, corpus):
        """
        Compute term frequency matrix with optimized memory usage.
        """
        def process_doc(doc):
            local_data = []
            local_indices = []
            counts = Counter(doc)
            for word, count in counts.items():
                vocab_index = self.vocab.get(word)
                if vocab_index is not None:
                    local_indices.append(vocab_index)
                    local_data.append(float(count))
            return local_data, local_indices

        # Use optimal number of partitions
        num_partitions = min(32, max(4, self.num_docs // 1000))
        bag = db.from_sequence(corpus, npartitions=num_partitions)
        results = bag.map(process_doc).compute()

        # Optimize memory usage with appropriate data types
        data_delayed = [
            da.from_delayed(delayed(np.array)(row[0], dtype=np.float32),
                          shape=(len(row[0]),), dtype=np.float32)
            for row in results
        ]
        indices_delayed = [
            da.from_delayed(delayed(np.array)(row[1], dtype=np.int32),
                          shape=(len(row[1]),), dtype=np.int32)
            for row in results
        ]
        
        d_data = da.concatenate(data_delayed)
        d_indices = da.concatenate(indices_delayed)

        counts_array = da.array([row.shape[0] for row in data_delayed], dtype=np.int32)
        d_indptr = da.concatenate([da.zeros(1, dtype=np.int32), da.cumsum(counts_array)])

        self.vocab_size = len(self.vocab)
        return csr_matrix((d_data.compute(), d_indices.compute(), d_indptr.compute()),
                         shape=(self.num_docs, self.vocab_size),
                         dtype=np.float32)

    def _compute_idf(self):
        """
        Compute inverse document frequency with optimized memory usage.
        """
        df = self.tf_matrix.getnnz(axis=0).astype(np.float32)
        df = np.maximum(df, 1e-6)
        return np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

    def query(self, query_tokens: Union[str, List[str]], metadata_filter: Optional[Dict] = None, 
             top_k: int = 10, do_keyword: bool = True) -> List[Dict]:
        """
        Query the BM25 index with optimized performance.
        """
        query_tokens = query_tokens if isinstance(query_tokens, list) else query_tokens.split(" ")
        assert len(query_tokens) > 0 or metadata_filter, "Query tokens or metadata cannot be empty"

        # Optimize query token processing
        query_tokens = [
            self.stemmer.stem(token.lower())
            for token in query_tokens
            if not self.stopwords or token.lower() not in self.stopwords
        ]

        assert len(query_tokens) > 0 or metadata_filter, "Query tokens must include words beyond the provided stop-words."

        # Optimize query vector creation
        qvec = np.zeros(len(self.vocab), dtype=np.float32)
        for word in query_tokens:
            if word in self.vocab:
                qvec[self.vocab[word]] += 1

        scores = _retrieve_scores(self.eager_index, self.tf_matrix.indices, self.tf_matrix.indptr, qvec)

        if do_keyword:
            t_texts_lower = TypedList(self.texts_lower)
            t_keywords = TypedList([token.lower() for token in query_tokens])
            scores += _compute_keyword_scores(t_texts_lower, t_keywords)

        if metadata_filter:
            if self.metadata is None:
                raise ValueError("Metadata not found in the index.")
            
            # Optimize metadata filtering with vectorized operations
            metadata = self.metadata
            bag = db.from_sequence(range(self.num_docs), npartitions=min(32, max(4, self.num_docs // 1000)))

            def compute_mask(i):
                md = metadata[i]
                return sum(md.get(k) in v for k, v in metadata_filter.items()) if any(md.get(k) in v for k, v in metadata_filter.items()) else 0.0

            mask = np.array(bag.map(compute_mask).compute(), dtype=np.float32)
            scores *= mask

        # Optimize top-k selection
        top_indices = np.argsort(-scores)[:top_k]

        return [
            {"text": self.texts[i], "score": float(scores[i]), **self.metadata[i]}
            for i in top_indices
            if scores[i] > 0
        ]

    def save(self, filepath):
        """
        Save the BM25 index state using joblib with gzip compression.
        """
        # Create a copy of state dict excluding non-serializable objects
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
            'idf': self.idf,
            'metadata': self.metadata,
            'texts': self.texts,
            'texts_lower': self.texts_lower,
            '_method_code': self._method_code,
            'eager_index': self.eager_index,
            'verbose': self.verbose
        }
        # Explicitly exclude lock and stemmer
        joblib.dump(state, filepath, compress=('gzip', 3))
        save_npz(filepath + "_tf_matrix.npz", self.tf_matrix)

    @staticmethod
    def load(filepath):
        """
        Load the BM25 index state using joblib with gzip decompression.
        """
        state = joblib.load(filepath)
        obj = BM25.__new__(BM25)  # Create an uninitialized BM25 instance.
        obj.__dict__.update(state)
        obj.tf_matrix = load_npz(filepath + "_tf_matrix.npz")
        # Recreate non-serializable attributes.
        obj.stemmer = PorterStemmer()
        obj.verbose = False
        obj.lock = Lock()
        return obj

    def save_hdf5(self, filepath):
        """
        Save the BM25 index using HDF5.
        Numeric data are stored as datasets, and non-numeric objects are
        pickled and stored as byte arrays.

        Args:
            filepath (str): Path where the HDF5 file will be saved

        Note:
            Non-serializable objects (lock, stemmer) are excluded from saving
            and recreated during load.
        """
        # Create a dictionary of attributes to save
        attrs_dict = {
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "variant": self.variant,
            "num_docs": self.num_docs,
            "_method_code": self._method_code
        }

        # Create dictionaries for numeric arrays and pickled objects
        numeric_arrays = {
            "doc_lengths": self.doc_lengths,
            "avgdl": np.array([self.avgdl]),
            "idf": self.idf,
            "eager_index": self.eager_index
        }

        pickle_objects = {
            "vocab": self.vocab,
            "metadata": self.metadata,
            "texts": self.texts,
        }

        with h5py.File(filepath, "w") as f:
            # Save attributes
            for key, value in attrs_dict.items():
                f.attrs[key] = value

            # Save numeric arrays
            for key, value in numeric_arrays.items():
                f.create_dataset(key, data=value)

            # Save pickled objects
            for key, value in pickle_objects.items():
                f.create_dataset(key, data=np.void(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)))

            # Save stopwords as string array
            stopwords_list = list(self.stopwords)
            f.create_dataset("stopwords", data=np.array(stopwords_list, dtype="S"))

        # Save sparse matrix separately
        save_npz(filepath + "_tf_matrix.npz", self.tf_matrix)

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
            obj.eager_index = f["eager_index"][:]

            # Restore Python objects from pickled data.
            obj.vocab = pickle.loads(bytes(f["vocab"][()]))
            obj.metadata = pickle.loads(bytes(f["metadata"][()]))
            obj.texts = pickle.loads(bytes(f["texts"][()]))
            obj.texts_lower = [l for l in map(str.lower, obj.texts)]

            # Restore stopwords (convert from bytes to string).
            obj.stopwords = set(s.decode('utf-8') for s in f["stopwords"][:])

            # Load sparse matrix
            obj.tf_matrix = load_npz(filepath + "_tf_matrix.npz")

            # Recreate any non-serializable attributes.
            obj.stemmer = PorterStemmer()
            obj.verbose = False
            obj.do_stem = False
            obj.lock = Lock()
            return obj


