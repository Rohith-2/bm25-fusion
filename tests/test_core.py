"""
Test cases for BM25 Fusion.
"""

import os
import tempfile
import pytest
import numpy as np
from bm25_fusion import BM25, tokenize, tokenize_texts


def test_bm25_query():
    """
    Test BM25 query with metadata filter.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25 = BM25(
        metadata=metadata,
        texts=corpus,
        variant="bm25",
        stopwords={"is", "a", "the", "and"},
    )
    results = bm25.query(["machine"], metadata_filter={"category": "science"}, top_k=2)
    assert len(results) >= 1
    for res in results:
        assert res.get("category") == "science"


def test_bm25_no_stopwords():
    """
    Test BM25 query without stopwords.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    bm25 = BM25(texts=corpus, variant="bm25")
    results = bm25.query(["machine"], top_k=2)
    assert len(results) > 0
    assert any("machine" in res["text"] for res in results)


def test_bm25_with_stopwords():
    """
    Test BM25 query with stopwords.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    bm25 = BM25(texts=corpus, variant="bm25", stopwords={"is", "a", "the", "and"})
    results = bm25.query(["learning"], top_k=2)
    assert len(results) > 0
    assert any("learning" in res["text"] for res in results)


def test_bm25_empty_query():
    """
    Test BM25 query with empty query raises ValueError.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    bm25 = BM25(texts=corpus, variant="bm25")
    with pytest.raises(ValueError) as excinfo:
        bm25.query([], top_k=2)
    assert "Query tokens and metadata_filter cannot both be empty" in str(excinfo.value)


def test_bm25_metadata_filter():
    """
    Test BM25 query with metadata filter.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25 = BM25(metadata=metadata, texts=corpus, variant="bm25")
    results = bm25.query(["hello"], metadata_filter={"category": "news"}, top_k=2)
    assert len(results) > 0
    for res in results:
        assert res.get("category") == "news"


def test_invalid_variant():
    """
    Test that using an invalid BM25 variant raises ValueError.
    """
    corpus = ["invalid variant test"]
    with pytest.raises(ValueError) as excinfo:
        BM25(texts=corpus, variant="unknown_variant")
    assert "Unknown BM25 variant" in str(excinfo.value)


def test_save_and_load(tmp_path):
    """
    Test saving and loading a BM25 index via Joblib Gzip.
    """
    corpus = ["save and load test", "another document"]
    metadata = [{"tag": "test"}, {"tag": "sample"}]
    bm25 = BM25(metadata=metadata, texts=corpus, variant="bm25")

    temp_file = tmp_path / "bm25_state.pkl.gz"
    bm25.save(str(temp_file))
    loaded_bm25 = BM25.load(str(temp_file))

    assert np.allclose(bm25.idf, loaded_bm25.idf)
    assert bm25.vocab == loaded_bm25.vocab
    assert loaded_bm25.texts == corpus
    assert loaded_bm25.metadata == metadata


def test_save_and_load_hdf5(tmp_path):
    """
    Test saving and loading a BM25 index via HDF5 binary store.
    """
    corpus = ["save and load test", "another document"]
    metadata = [{"tag": "test"}, {"tag": "sample"}]
    bm25 = BM25(metadata=metadata, texts=corpus, variant="bm25")

    temp_file = tmp_path / "bm25_state.h5"
    bm25.save_hdf5(str(temp_file))
    loaded_bm25 = BM25.load_hdf5(str(temp_file))

    assert np.allclose(bm25.idf, loaded_bm25.idf)
    assert bm25.vocab == loaded_bm25.vocab
    assert loaded_bm25.texts == corpus
    assert loaded_bm25.metadata == metadata


def test_remove_document():
    """
    Test successful document removal and failure for non-existent document.
    """
    corpus = ["doc one", "doc two"]
    bm25 = BM25(texts=corpus, variant="bm25")
    bm25.remove_document("doc one")
    assert bm25.num_docs == 1
    assert bm25.texts == ["doc two"]

    with pytest.raises(ValueError) as excinfo:
        bm25.remove_document("non-existent document")
    assert "not found in index" in str(excinfo.value)


def test_query_all_stopwords():
    """
    Test query where all tokens are stopwords.
    """
    corpus = ["all stopwords test"]
    bm25 = BM25(texts=corpus, variant="bm25", stopwords={"all", "stopwords", "test"})
    with pytest.raises(ValueError) as excinfo:
        bm25.query(["all", "stopwords", "test"], top_k=2)
    assert "must contain words beyond stop-words" in str(excinfo.value)


def test_query_token_not_in_vocab():
    """
    Test querying with tokens that are not in the vocabulary.
    """
    corpus = ["the quick brown fox", "jumps over the lazy dog"]
    bm25 = BM25(texts=corpus, variant="bm25")
    results = bm25.query(["nonexistent"], top_k=2)
    assert results == []


def test_add_document_behavior():
    """
    Test adding single and batch documents to the BM25 index.
    """
    corpus = ["initial document"]
    bm25 = BM25(texts=corpus, variant="bm25")

    # Add single document string
    bm25.add_document("new single document", new_metadata={"added": True})
    assert bm25.num_docs == 2
    assert bm25.metadata[1] == {"added": True}

    # Add batch documents list with None metadata
    bm25.add_document(["batch doc 1", "batch doc 2"])
    assert bm25.num_docs == 4
    assert len(bm25.metadata) == 4


def test_idf_vector_shape():
    """
    Verify IDF is a per-term vector matching vocabulary size.
    """
    corpus = ["apple banana", "apple orange", "banana orange cherry"]
    bm25 = BM25(texts=corpus, variant="bm25")
    assert bm25.idf.ndim == 1
    assert len(bm25.idf) == len(bm25.vocab)


def test_variants():
    """
    Test that all supported BM25 variants initialize and query without error.
    """
    corpus = ["hello world", "foo bar baz"]
    for variant in ["bm25", "bm25+", "bm25l", "atire"]:
        bm25 = BM25(texts=corpus, variant=variant)
        res = bm25.query("hello", top_k=1)
        assert len(res) > 0


if __name__ == "__main__":
    pytest.main()
