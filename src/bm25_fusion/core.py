import threading
import pickle
from collections import defaultdict
import jax.numpy as jnp
from tokenization import tokenize 
from retrieval import optimized_query

class BM25:
    def __init__(self, texts=None, metadata=None, k1=1.5, b=0.75, delta=0.5,
                 variant='bm25', stopwords=None):
        """
        Initialize BM25 instance.
        
        Parameters:
          texts: list of document strings.
          metadata: list of dicts corresponding to each document.
          k1, b, delta: BM25 parameters.
          variant: one of {"bm25", "bm25+", "bm25l", "atire"}.
          stopwords: iterable of stopwords to filter out; if None, no filtering is done.
        """
        self.k1, self.b, self.delta = k1, b, delta
        self.variant = variant.lower()
        self.texts = texts if texts else []
        # Tokenize each text and filter stopwords (case-insensitive)
        if texts:
            self.corpus_tokens = []
            for doc in texts:
                tokens = tokenize(doc)
                if stopwords:
                    tokens = [t for t in tokens if t.lower() not in {s.lower() for s in stopwords}]
                self.corpus_tokens.append(tokens)
        else:
            self.corpus_tokens = []
        self.num_docs = len(self.corpus_tokens)
        self.doc_lengths = jnp.array([len(doc) for doc in self.corpus_tokens], dtype=jnp.float32) if self.num_docs > 0 else jnp.array([])
        self.avgdl = jnp.mean(self.doc_lengths) if self.num_docs > 0 else 1.0
        
        # Build vocabulary and inverted index
        self.vocab, self.inverted_index = self._build_index(self.corpus_tokens)
        # Compute idf for each term (using classic BM25 formula)
        self.idf = self._compute_idf()
        # For ATIRE, clamp idf to nonnegative values.
        if self.variant == "atire":
            self.idf = {term: jnp.maximum(val, 0) for term, val in self.idf.items()}
        self.metadata = metadata if metadata else [{} for _ in range(self.num_docs)]
        self.lock = threading.Lock()
    
    def _build_index(self, corpus):
        """Build vocabulary and inverted index from corpus_tokens."""
        vocab = {}
        inverted_index = defaultdict(list)
        for doc_id, doc in enumerate(corpus):
            term_freqs = defaultdict(int)
            for word in doc:
                if word not in vocab:
                    vocab[word] = len(vocab)
                term_freqs[word] += 1
            for term, freq in term_freqs.items():
                inverted_index[term].append((doc_id, freq))
        return vocab, inverted_index

    def _compute_idf(self):
        """Compute idf for each term using the formula: log((N - df + 0.5)/(df + 0.5) + 1)."""
        df = {term: len(postings) for term, postings in self.inverted_index.items()}
        idf = {}
        for term, count in df.items():
            idf[term] = jnp.log((self.num_docs - count + 0.5) / (count + 0.5) + 1)
        return idf

    def _compute_term_score(self, tf, idf_val, doc_len):
        """Compute BM25 score for a term in a document based on variant."""
        norm = self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
        if self.variant in ("bm25", "atire"):
            return idf_val * ((tf * (self.k1 + 1)) / (tf + norm))
        elif self.variant == "bm25+":
            return idf_val * (((tf + self.delta) * (self.k1 + 1)) / (tf + norm + self.delta))
        elif self.variant == "bm25l":
            return idf_val * (tf / (tf + norm + self.delta * (doc_len / self.avgdl)))
        else:
            return 0.0
        
    def query(self, query, top_k=10, metadata_filter=None,threshold=0.01):
        """
        Retrieve documents matching the query.
        
        Parameters:
          query: query string.
          top_k: number of top documents to return.
          metadata_filter: optional dict for metadata filtering.
          do_keyword: if True, add bonus score based on raw keyword matching.
        
        Returns: list of dicts with keys "text" and "score" (plus metadata fields).
        """
        return optimized_query(self.num_docs, self.doc_lengths, self.avgdl, self.inverted_index, 
                               self.idf, self.k1, self.b, query, self.texts, self.metadata, metadata_filter, 
                               top_k, self.variant, self.delta,threshold)

    def add_document(self, new_texts, new_metadata=None):
        """
        Add new document(s) to the index.
        new_texts: list of document strings.
        new_metadata: list of metadata dicts, one per document.
        """
        with self.lock:
            for idx, text in enumerate(new_texts):
                doc_id = self.num_docs
                self.texts.append(text)
                tokens = tokenize(text)
                self.corpus_tokens.append(tokens)
                self.num_docs += 1
                # Update doc_lengths and average doc length
                doc_len = len(tokens)
                self.doc_lengths = jnp.concatenate([self.doc_lengths, jnp.array([doc_len], dtype=jnp.float32)])
                self.avgdl = jnp.mean(self.doc_lengths)
                # Update vocabulary and inverted index for this document
                term_freqs = defaultdict(int)
                for word in tokens:
                    term_freqs[word] += 1
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
                for term, freq in term_freqs.items():
                    self.inverted_index[term].append((doc_id, freq))
                # Update metadata
                if new_metadata and idx < len(new_metadata):
                    self.metadata.append(new_metadata[idx])
                else:
                    self.metadata.append({})
            # Recompute idf
            self.idf = self._compute_idf()
            if self.variant == "atire":
                self.idf = {term: jnp.maximum(val, 0) for term, val in self.idf.items()}

    def remove_document(self, text):
        """
        Remove the first document matching the given text.
        """
        with self.lock:
            for i, doc_text in enumerate(self.texts):
                if doc_text == text:
                    doc_id = i
                    # Remove postings corresponding to doc_id in inverted index
                    for term in list(self.inverted_index.keys()):
                        self.inverted_index[term] = [(d, tf) for d, tf in self.inverted_index[term] if d != doc_id]
                    # Remove document data
                    del self.texts[doc_id]
                    del self.corpus_tokens[doc_id]
                    # Rebuild doc_lengths and update average length
                    self.doc_lengths = jnp.array([len(doc) for doc in self.corpus_tokens], dtype=jnp.float32)
                    self.avgdl = jnp.mean(self.doc_lengths) if self.num_docs > 1 else 1.0
                    del self.metadata[doc_id]
                    self.num_docs -= 1
                    # Recompute idf
                    self.idf = self._compute_idf()
                    if self.variant == "atire":
                        self.idf = {term: jnp.maximum(val, 0) for term, val in self.idf.items()}
                    break

    def save_index(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.vocab, dict(self.inverted_index), self.idf,
                         self.doc_lengths, self.avgdl, self.texts,
                         self.corpus_tokens, self.metadata), f)

    @staticmethod
    def load_index(path):
        with open(path, 'rb') as f:
            vocab, inverted_index, idf, doc_lengths, avgdl, texts, corpus_tokens, metadata = pickle.load(f)
            instance = BM25(texts=texts, metadata=metadata)
            instance.vocab = vocab
            instance.inverted_index = defaultdict(list, inverted_index)
            instance.idf = idf
            instance.doc_lengths = doc_lengths
            instance.avgdl = avgdl
            instance.corpus_tokens = corpus_tokens
            instance.num_docs = len(texts)
            return instance
