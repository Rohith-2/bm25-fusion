"""
Retrieval module for BM25 Fusion.
"""

from bm25_fusion.core import BM25
from bm25_fusion.tokenization import tokenize

def build_index(corpus_docs, metadata_docs=None, stopwords=None, variant="bm25", delta=0.5):
    """
    Builds a BM25 index from a list of text documents.
    """
    corpus_tokens = [tokenize(doc) for doc in corpus_docs]
    return BM25(corpus_tokens, metadata=metadata_docs, texts=corpus_docs, variant=variant,
                delta=delta, stopwords=stopwords)

def run_query(bm25_index, query_text, metadata_filter=None, top_k=10):
    """
    Tokenizes the query and returns BM25 search results.
    """
    query_tokens = tokenize(query_text)
    return bm25_index.query(query_tokens, metadata_filter=metadata_filter, top_k=top_k)

if __name__ == "__main__":
    # Example usage:
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25_idx = build_index(corpus, metadata_docs=metadata, stopwords={"is", "a", "the", "and"})
    results = run_query(bm25_idx, "machine learning",
                        metadata_filter={"category": "science"}, top_k=2)
    print(results)
