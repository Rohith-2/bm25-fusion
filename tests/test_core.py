import pytest
from bm25_fusion.core import BM25
from bm25_fusion.tokenization import tokenize

def test_bm25_query():
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    corpus_tokens = [tokenize(doc) for doc in corpus]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25 = BM25(corpus_tokens, metadata=metadata, texts=corpus, variant="bm25", stopwords={"is", "a", "the", "and"})
    # Query for "machine"
    results = bm25.query(["machine"], metadata_filter={"category": "science"}, top_k=2)
    assert len(results) >= 0
    for res in results:
        # Check that metadata filter is satisfied.
        assert res.get("category") == "science"

def test_bm25_no_stopwords():
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    corpus_tokens = [doc.split() for doc in corpus]
    bm25 = BM25(corpus_tokens,texts=corpus, variant="bm25")
    # Query for "machine"
    results = bm25.query(["machine"], top_k=2)
    assert len(results) >= 0
    assert any("machine" in res["text"] for res in results)

def test_bm25_with_stopwords():
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    corpus_tokens = [doc.split() for doc in corpus]
    bm25 = BM25(corpus_tokens,texts=corpus, variant="bm25", stopwords={"is", "a", "the", "and"})
    # Query for "learning"
    results = bm25.query(["learning"], top_k=2)
    assert len(results) >= 0
    assert any("learning" in res["text"] for res in results)

def test_bm25_empty_query():
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    corpus_tokens = [doc.split() for doc in corpus]
    bm25 = BM25(corpus_tokens,texts=corpus, variant="bm25")
    # Query with empty list
    results = bm25.query([], top_k=2)
    assert len(results) == 0

def test_bm25_metadata_filter():
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    corpus_tokens = [doc.split() for doc in corpus]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25 = BM25(corpus_tokens, metadata=metadata, texts=corpus, variant="bm25")
    # Query for "hello" with metadata filter
    results = bm25.query(["hello"], metadata_filter={"category": "news"}, top_k=2)
    assert len(results) >= 0
    for res in results:
        assert res.get("category") == "news"

if __name__ == "__main__":
    pytest.main()