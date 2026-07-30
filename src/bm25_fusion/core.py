"""
Core BM25 engine module for BM25 Fusion.
Provides eager-indexed BM25 retrieval with Numba JIT acceleration, metadata filtering,
dynamic index updating, and dual persistence formats (Joblib Gzip & HDF5).
"""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-positional-arguments

import gc
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import h5py
import joblib
import numpy as np
from nltk.stem import PorterStemmer
from numba import njit, prange
from numba.typed import List as TypedList
from tqdm import tqdm

from .tokenization import tokenize_texts

# Standard fallback stopwords set if NLTK corpora are unavailable offline
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "couldn't", "did", "didn't",
    "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "hed", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
}


def _get_default_stopwords() -> Set[str]:
    """
    Safely load default English stopwords from NLTK with static set fallback.
    """
    try:
        from nltk.corpus import stopwords as st
        return set(s.lower() for s in st.words("english"))
    except Exception:
        return set(DEFAULT_STOPWORDS)


class BM25:
    """
    High-performance BM25 retrieval engine with eager sparse index precomputation,
    metadata filtering, real-time index mutation, and Numba acceleration.
    """

    def __init__(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        variant: str = "bm25",
        stopwords: Optional[Iterable[str]] = None,
        do_stem: bool = False,
        stemmer: Optional[Any] = None,
        num_processes: int = 4,
        show_progress: bool = True,
    ) -> None:
        """
        Initializes a BM25 index instance over a document corpus.

        Args:
            texts: List of raw document strings.
            metadata: Optional list of dictionaries containing metadata for each document.
            k1: BM25 term frequency saturation parameter k1.
            b: BM25 document length normalization parameter b.
            delta: Delta parameter used in BM25+ and BM25L variants.
            variant: Scoring variant ("bm25", "lucene", "robertson", "bm25+", "bm25l", "atire").
            stopwords: Optional custom set/iterable of stopword strings.
            do_stem: Whether to apply Porter stemming to corpus tokens during build.
            stemmer: Optional stemmer instance (defaults to PorterStemmer).
            num_processes: Parallel process workers for tokenization.
            show_progress: Display progress bar during construction.
        """
        if texts is None or not isinstance(texts, list):
            raise TypeError("texts must be a non-null list of document text strings.")
        if len(texts) == 0:
            raise ValueError("texts corpus cannot be empty.")

        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self.variant = str(variant).lower()
        self.num_docs = len(texts)
        self.texts = list(texts)
        self.do_stem = bool(do_stem)
        self.stemmer = stemmer if stemmer is not None else PorterStemmer()

        if stopwords is not None:
            self.stopwords = set(str(s).lower() for s in stopwords)
        else:
            self.stopwords = _get_default_stopwords()

        if metadata is not None:
            if not isinstance(metadata, list) or len(metadata) != self.num_docs:
                raise ValueError("metadata must be a list of dicts equal in length to texts.")
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(self.num_docs)]

        # Precompute lower-case raw texts for fast keyword searching
        self.texts_lower = [t.lower() for t in self.texts]

        # Determine numerical method code for JIT dispatcher
        if self.variant in ("bm25", "lucene", "robertson"):
            self._method_code = 0
        elif self.variant == "bm25+":
            self._method_code = 1
        elif self.variant == "bm25l":
            self._method_code = 2
        elif self.variant == "atire":
            self._method_code = 3
        else:
            raise ValueError(
                f"Unknown BM25 variant '{variant}'. Supported: 'bm25', 'bm25+', 'bm25l', 'atire'."
            )

        # Initialize concurrency lock for live updates
        self.lock = Lock()

        # Tokenize and build sparse index representation
        corpus_tokens = tokenize_texts(
            self.texts, num_processes=num_processes, show_progress=show_progress
        )
        self.doc_lengths = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.num_docs > 0 else 0.0

        if self.do_stem:
            corpus_tokens = self._stem_corpus(corpus_tokens, show_progress=show_progress)

        self.vocab = self._build_vocab(corpus_tokens, show_progress=show_progress)
        self.vocab_size = len(self.vocab)
        self.tf_matrix = self._compute_tf_matrix(corpus_tokens, show_progress=show_progress)
        del corpus_tokens
        gc.collect()

        self.idf = self._compute_idf()
        if self._method_code == 3:  # ATIRE clamps negative IDF to 0
            self.idf = np.maximum(self.idf, 0.0)

        self.eager_index = _eager_scores(
            self.tf_matrix[0],
            self.tf_matrix[1],
            self.tf_matrix[2],
            self.idf,
            self.doc_lengths,
            self.avgdl,
            self._method_code,
            self.k1,
            self.b,
            self.delta,
        )

    def _stem_corpus(
        self, corpus: List[List[str]], show_progress: bool = True
    ) -> List[List[str]]:
        """
        Applies stemming across corpus tokens in parallel.
        """
        def stem_doc(doc: List[str]) -> List[str]:
            return [self.stemmer.stem(word) for word in doc]

        with ThreadPoolExecutor() as executor:
            stemmed = list(
                executor.map(
                    stem_doc,
                    tqdm(corpus, total=len(corpus), disable=not show_progress),
                )
            )
        return stemmed

    def _build_vocab(
        self, corpus: List[List[str]], show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Builds vocabulary word-to-index mapping from tokenized corpus.
        """
        def unique_words(doc: List[str]) -> Set[str]:
            return set(doc)

        with ThreadPoolExecutor() as executor:
            sets = list(
                executor.map(
                    unique_words,
                    tqdm(corpus, total=len(corpus), disable=not show_progress),
                )
            )

        unique_words_set = set().union(*sets) if sets else set()
        return {word: i for i, word in enumerate(sorted(unique_words_set))}

    def _compute_tf_matrix(
        self, corpus: List[List[str]], show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes term frequency CSR sparse matrix structure.

        Returns:
            Tuple of (tf_data, tf_indices, tf_indptr) numpy arrays.
        """
        data_list: List[float] = []
        indices_list: List[int] = []
        indptr: List[int] = [0]

        iterator = tqdm(corpus, total=len(corpus), disable=not show_progress)
        for doc in iterator:
            counts = Counter(doc)
            for word, count in counts.items():
                vocab_index = self.vocab.get(word)
                if vocab_index is not None:
                    indices_list.append(vocab_index)
                    data_list.append(float(count))
            indptr.append(len(data_list))

        return (
            np.array(data_list, dtype=np.float32),
            np.array(indices_list, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
        )

    def _compute_idf(self) -> np.ndarray:
        """
        Computes per-term Inverse Document Frequency (IDF) vector across vocabulary.
        Uses fast vectorized bincount over sparse term indices.
        """
        if self.vocab_size == 0 or self.num_docs == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)

        # Count document frequency for each term index present in tf_matrix[1]
        df = np.bincount(self.tf_matrix[1], minlength=self.vocab_size).astype(np.float32)
        df = np.maximum(df, 1.0)
        idf = np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        return idf

    def query(
        self,
        query_tokens: Union[str, List[str]],
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        do_keyword: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Queries the BM25 index with text or token list, applying optional metadata filtering.

        Args:
            query_tokens: Query string or list of token strings.
            metadata_filter: Dictionary specifying metadata constraint filters.
            top_k: Maximum number of top search results to return.
            do_keyword: Perform exact substring keyword score boosting.

        Returns:
            List of result dictionaries containing text, score, and document metadata.
        """
        if isinstance(query_tokens, str):
            raw_tokens = [t for t in query_tokens.split() if t]
        elif isinstance(query_tokens, list):
            raw_tokens = query_tokens
        else:
            raise TypeError("query_tokens must be a string or list of token strings.")

        if not raw_tokens and not metadata_filter:
            raise ValueError("Query tokens and metadata_filter cannot both be empty.")

        processed_query: List[str] = []
        for token in raw_tokens:
            t_lower = token.lower()
            if self.stopwords and t_lower in self.stopwords:
                continue
            stemmed_t = self.stemmer.stem(t_lower) if self.do_stem else t_lower
            processed_query.append(stemmed_t)

        if not processed_query and not metadata_filter:
            raise ValueError("Query tokens must contain words beyond stop-words.")

        query_mask = np.zeros(self.vocab_size, dtype=np.bool_)
        for word in processed_query:
            if word in self.vocab:
                query_mask[self.vocab[word]] = True

        scores = _retrieve_scores(
            self.eager_index, self.tf_matrix[1], self.tf_matrix[2], query_mask
        )

        if do_keyword and processed_query:
            t_texts_lower = TypedList(self.texts_lower)
            t_keywords = TypedList([t.lower() for t in processed_query])
            scores += _compute_keyword_scores(t_texts_lower, t_keywords)

        if metadata_filter:
            normalized_filter: Dict[str, Set[Any]] = {}
            for k, v in metadata_filter.items():
                if isinstance(v, (list, tuple, set)):
                    normalized_filter[k] = set(v)
                else:
                    normalized_filter[k] = {v}

            mask = np.ones(self.num_docs, dtype=np.float32)
            for i in range(self.num_docs):
                doc_meta = self.metadata[i]
                match = True
                for k, allowed in normalized_filter.items():
                    if k not in doc_meta or doc_meta[k] not in allowed:
                        match = False
                        break
                if not match:
                    mask[i] = 0.0

            scores *= mask

        top_indices = np.argsort(-scores)[:top_k]

        results = [
            {"text": self.texts[i], "score": float(scores[i]), **self.metadata[i]}
            for i in top_indices
            if scores[i] > 0
        ]
        return results

    def save(self, filepath: str) -> None:
        """
        Saves the BM25 index state to disk using Joblib with gzip compression.
        """
        state = {
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "variant": self.variant,
            "stopwords": list(self.stopwords),
            "num_docs": self.num_docs,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "tf_matrix": self.tf_matrix,
            "idf": self.idf,
            "metadata": self.metadata,
            "texts": self.texts,
            "texts_lower": self.texts_lower,
            "_method_code": self._method_code,
            "eager_index": self.eager_index,
            "do_stem": self.do_stem,
        }
        joblib.dump(state, filepath, compress=("gzip", 3))

    @staticmethod
    def load(filepath: str) -> "BM25":
        """
        Loads a BM25 index instance from a Joblib compressed file.
        """
        state = joblib.load(filepath)
        obj = BM25.__new__(BM25)
        obj.__dict__.update(state)
        obj.stemmer = PorterStemmer()
        obj.lock = Lock()
        return obj

    def save_hdf5(self, filepath: str) -> None:
        """
        Saves the BM25 index to an HDF5 binary store.
        """
        with h5py.File(filepath, "w") as f:
            f.attrs["k1"] = self.k1
            f.attrs["b"] = self.b
            f.attrs["delta"] = self.delta
            f.attrs["variant"] = self.variant
            f.attrs["num_docs"] = self.num_docs
            f.attrs["_method_code"] = self._method_code
            f.attrs["do_stem"] = self.do_stem

            f.create_dataset("doc_lengths", data=self.doc_lengths)
            f.create_dataset("avgdl", data=np.array([self.avgdl]))
            f.create_dataset("idf", data=self.idf)
            f.create_dataset("tf_matrix_0", data=self.tf_matrix[0])
            f.create_dataset("tf_matrix_1", data=self.tf_matrix[1])
            f.create_dataset("tf_matrix_2", data=self.tf_matrix[2])
            f.create_dataset("eager_index", data=self.eager_index)

            f.create_dataset(
                "vocab",
                data=np.void(pickle.dumps(self.vocab, protocol=pickle.HIGHEST_PROTOCOL)),
            )
            f.create_dataset(
                "metadata",
                data=np.void(pickle.dumps(self.metadata, protocol=pickle.HIGHEST_PROTOCOL)),
            )
            f.create_dataset(
                "texts",
                data=np.void(pickle.dumps(self.texts, protocol=pickle.HIGHEST_PROTOCOL)),
            )
            f.create_dataset(
                "texts_lower",
                data=np.void(pickle.dumps(self.texts_lower, protocol=pickle.HIGHEST_PROTOCOL)),
            )

            stopwords_list = list(self.stopwords)
            f.create_dataset("stopwords", data=np.array(stopwords_list, dtype="S"))

    @staticmethod
    def load_hdf5(filepath: str) -> "BM25":
        """
        Loads a BM25 index instance from an HDF5 binary store.
        """
        with h5py.File(filepath, "r") as f:
            obj = BM25.__new__(BM25)
            obj.k1 = float(f.attrs["k1"])
            obj.b = float(f.attrs["b"])
            obj.delta = float(f.attrs["delta"])
            obj.variant = str(f.attrs["variant"])
            obj.num_docs = int(f.attrs["num_docs"])
            obj._method_code = int(f.attrs["_method_code"])
            obj.do_stem = bool(f.attrs.get("do_stem", False))

            obj.doc_lengths = f["doc_lengths"][:]
            obj.avgdl = float(f["avgdl"][0])
            obj.idf = f["idf"][:]

            tf0 = f["tf_matrix_0"][:]
            tf1 = f["tf_matrix_1"][:]
            tf2 = f["tf_matrix_2"][:]
            obj.tf_matrix = (tf0, tf1, tf2)
            obj.eager_index = f["eager_index"][:]

            obj.vocab = pickle.loads(bytes(f["vocab"][()]))
            obj.vocab_size = len(obj.vocab)
            obj.metadata = pickle.loads(bytes(f["metadata"][()]))
            obj.texts = pickle.loads(bytes(f["texts"][()]))
            obj.texts_lower = pickle.loads(bytes(f["texts_lower"][()]))

            obj.stopwords = set(s.decode("utf-8") for s in f["stopwords"][:])

            obj.stemmer = PorterStemmer()
            obj.lock = Lock()
            return obj

    def _rebuild_index(self, num_processes: int = 4) -> None:
        """
        Rebuilds the BM25 precomputed structures from current texts state.
        """
        tokenized_texts = tokenize_texts(
            self.texts, num_processes=num_processes, show_progress=False
        )
        self.doc_lengths = np.array([len(doc) for doc in tokenized_texts], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.doc_lengths.size > 0 else 0.0

        if self.do_stem:
            tokenized_texts = self._stem_corpus(tokenized_texts, show_progress=False)

        self.vocab = self._build_vocab(tokenized_texts, show_progress=False)
        self.vocab_size = len(self.vocab)
        self.tf_matrix = self._compute_tf_matrix(tokenized_texts, show_progress=False)
        del tokenized_texts
        gc.collect()

        self.idf = self._compute_idf()
        if self._method_code == 3:
            self.idf = np.maximum(self.idf, 0.0)

        self.eager_index = _eager_scores(
            self.tf_matrix[0],
            self.tf_matrix[1],
            self.tf_matrix[2],
            self.idf,
            self.doc_lengths,
            self.avgdl,
            self._method_code,
            self.k1,
            self.b,
            self.delta,
        )

    def add_document(
        self,
        new_text: Union[str, List[str]],
        new_metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        num_processes: int = 4,
    ) -> None:
        """
        Adds document(s) to the index and rebuilds the BM25 eager score index.

        Args:
            new_text: Single text string or list of text strings.
            new_metadata: Optional single metadata dict or list of dicts.
            num_processes: Worker process count for tokenization rebuild.
        """
        if isinstance(new_text, str):
            doc_list = [new_text]
        elif isinstance(new_text, list):
            doc_list = new_text
        else:
            raise TypeError("new_text must be a string or list of text strings.")

        if not doc_list:
            return

        if new_metadata is None:
            meta_list = [{} for _ in range(len(doc_list))]
        elif isinstance(new_metadata, dict):
            if len(doc_list) != 1:
                raise ValueError("Single metadata dictionary provided for multiple documents.")
            meta_list = [new_metadata]
        elif isinstance(new_metadata, list):
            if len(new_metadata) != len(doc_list):
                raise ValueError("Length of new_metadata must match length of new_text list.")
            meta_list = new_metadata
        else:
            raise TypeError("new_metadata must be a dict or list of dicts.")

        with self.lock:
            self.texts.extend(doc_list)
            self.texts_lower.extend([t.lower() for t in doc_list])
            self.metadata.extend(meta_list)
            self.num_docs += len(doc_list)
            self._rebuild_index(num_processes=num_processes)

    def remove_document(self, text: str) -> None:
        """
        Removes the first document matching the provided text and rebuilds index.

        Args:
            text: Exact text of the document to remove.
        """
        with self.lock:
            try:
                idx = self.texts.index(text)
            except ValueError as e:
                raise ValueError(f"Document matching '{text}' not found in index.") from e

            del self.texts[idx]
            del self.texts_lower[idx]
            del self.metadata[idx]
            self.num_docs -= 1
            self._rebuild_index()


@njit(parallel=True)
def _eager_scores(
    tf_data: np.ndarray,
    tf_indices: np.ndarray,
    tf_indptr: np.ndarray,
    idf: np.ndarray,
    doc_lengths: np.ndarray,
    avgdl: float,
    method_code: int,
    k1: float,
    b: float,
    delta: float,
) -> np.ndarray:
    num_docs = len(doc_lengths)
    score_data = np.empty_like(tf_data)
    if avgdl <= 0.0:
        return score_data

    for d in prange(num_docs):
        norm = k1 * (1.0 - b + b * doc_lengths[d] / avgdl)
        for j in range(tf_indptr[d], tf_indptr[d + 1]):
            tf = tf_data[j]
            term_idx = tf_indices[j]
            term_idf = idf[term_idx]

            if method_code in (0, 3):
                score = term_idf * ((tf * (k1 + 1.0)) / (tf + norm))
            elif method_code == 1:
                score = term_idf * (((tf + delta) * (k1 + 1.0)) / (tf + norm + delta))
            elif method_code == 2:
                score = term_idf * (tf / (tf + norm + delta * (doc_lengths[d] / avgdl)))
            else:
                score = 0.0

            score_data[j] = score

    return score_data


@njit(parallel=True)
def _retrieve_scores(
    eager_data: np.ndarray,
    tf_indices: np.ndarray,
    tf_indptr: np.ndarray,
    query_mask: np.ndarray,
) -> np.ndarray:
    num_docs = len(tf_indptr) - 1
    scores = np.zeros(num_docs, dtype=np.float32)

    for d in prange(num_docs):
        s = 0.0
        for j in range(tf_indptr[d], tf_indptr[d + 1]):
            idx = tf_indices[j]
            if query_mask[idx]:
                s += eager_data[j]
        scores[d] = s

    return scores


@njit(parallel=True)
def _compute_keyword_scores(texts: List[str], keywords: List[str]) -> np.ndarray:
    num_docs = len(texts)
    keyword_scores = np.zeros(num_docs, dtype=np.float32)
    num_keywords = len(keywords)

    if num_keywords == 0:
        return keyword_scores

    for i in prange(num_docs):
        doc_text = texts[i]
        for k_idx in range(num_keywords):
            if keywords[k_idx] in doc_text:
                keyword_scores[i] += 1.0

    return keyword_scores

