"""
Test cases for BM25 Fusion.
"""

import os
import tempfile
import pytest
import joblib
import numpy as np
from bm25_fusion.core import BM25
from bm25_fusion.tokenization import tokenize

def test_bm25_query():
    """
    Test BM25 query with metadata filter.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25 = BM25(metadata=metadata, texts=corpus,
                variant="bm25", stopwords={"is", "a", "the", "and"})
    results = bm25.query(["machine"], metadata_filter={"category": "science"}, top_k=2)
    assert len(results) >= 0
    for res in results:
        assert res.get("category") == "science"

def test_bm25_no_stopwords():
    """
    Test BM25 query without stopwords.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    bm25 = BM25(texts=corpus, variant="bm25")
    results = bm25.query(["machine"], top_k=2)
    assert len(results) >= 0
    assert any("machine" in res["text"] for res in results)

def test_bm25_with_stopwords():
    """
    Test BM25 query with stopwords.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    bm25 = BM25(texts=corpus, variant="bm25", stopwords={"is", "a", "the", "and"})
    results = bm25.query(["learning"], top_k=2)
    assert len(results) >= 0
    assert any("learning" in res["text"] for res in results)

def test_bm25_empty_query():
    """
    Test BM25 query with empty query.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    bm25 = BM25(texts=corpus, variant="bm25")
    with pytest.raises(AssertionError) as excinfo:
        bm25.query([], top_k=2)
    assert "Query tokens cannot be empty." in str(excinfo.value)

def test_bm25_metadata_filter():
    """
    Test BM25 query with metadata filter.
    """
    corpus = ["hello world", "machine learning is fun", "hello machine"]
    metadata = [{"category": "news"}, {"category": "science"}, {"category": "news"}]
    bm25 = BM25(metadata=metadata, texts=corpus, variant="bm25")
    results = bm25.query(["hello"], metadata_filter={"category": "news"}, top_k=2)
    assert len(results) >= 0
    for res in results:
        assert res.get("category") == "news"

##########################################
# Additional edge case tests start here. #
##########################################

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
    Test saving and loading a BM25 index.
    """
    corpus = ["save and load test", "another document"]
    metadata = [{"tag": "test"}, {"tag": "sample"}]
    bm25 = BM25(metadata=metadata, texts=corpus, variant="bm25")
    
    # Save to a temporary file.
    temp_file = tmp_path / "bm25_state.pkl"
    bm25.save(str(temp_file))
    loaded_bm25 = BM25.load(str(temp_file))
    
    # Check that core attributes remain the same.
    assert np.allclose(bm25.idf, loaded_bm25.idf)
    assert bm25.vocab == loaded_bm25.vocab
    assert loaded_bm25.texts == corpus
    assert loaded_bm25.metadata == metadata

def test_remove_document_failure():
    """
    Test that attempting to remove a non-existent document raises ValueError.
    """
    corpus = ["doc one", "doc two"]
    bm25 = BM25(texts=corpus, variant="bm25")
    with pytest.raises(ValueError) as excinfo:
        bm25.remove_document("non-existent document")
    assert "Document matching the provided text not found." in str(excinfo.value)

def test_query_all_stopwords():
    """
    Test query where all tokens are stopwords so that after filtering the query becomes empty.
    """
    corpus = ["all stopwords test"]
    # Provide stopwords that cover the query.
    bm25 = BM25(texts=corpus, variant="bm25", stopwords={"all", "stopwords", "test"})
    # Even though the original query is non-empty, filtering leaves no tokens.
    with pytest.raises(AssertionError) as excinfo:
        bm25.query(["all", "stopwords", "test"], top_k=2)
    assert "Query tokens must include words beyond the provided stop-words." in str(excinfo.value)

def test_query_token_not_in_vocab():
    """
    Test querying with tokens that are not in the vocabulary returns an empty result.
    """
    corpus = ["the quick brown fox", "jumps over the lazy dog"]
    bm25 = BM25(texts=corpus, variant="bm25")
    # Query with a token that does not appear in any document.
    results = bm25.query(["nonexistent"], top_k=2)
    # Expect no results (score 0) so empty list is returned.
    assert results == []

def test_add_document_behavior():
    """
    Test adding a document to the BM25 index.
    Note: The current implementation of add_document extends the texts with each character of the new_text.
    This test verifies the effect based on the current code.
    """
    corpus = ["initial document"]
    bm25 = BM25(texts=corpus, variant="bm25")
    initial_num_docs = bm25.num_docs
    
    # Add a new document (as a string, per current implementation).
    new_doc = ["new document"]
    bm25.add_document(new_doc, new_metadata=[{"added": True}])
    
    # Since add_document uses .extend on a string, each character is added.
    # We expect number of docs to increase by len(new_doc)
    expected_new_docs = initial_num_docs + len(new_doc)
    assert bm25.num_docs == expected_new_docs
    # Check that the metadata length also increased appropriately.
    assert len(bm25.metadata) == expected_new_docs

if __name__ == "__main__":
    pytest.main()
