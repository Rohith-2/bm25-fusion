"""
Retrieval module for BM25 Fusion using JAX.
"""

import jax.numpy as jnp

def _compute_keyword_scores(texts, keywords):
    """
    Compute keyword match scores for each document.
    texts: list of strings
    keywords: list of query tokens (strings)
    Returns: jnp.array of shape (num_docs,) of float32 scores
    """
    # Convert texts to lowercase
    texts_lower = [text.lower() for text in texts]
    # Create a mask: for each keyword, check if it's in each text
    keyword_masks = [[keyword.lower() in text for text in texts_lower] for keyword in keywords]
    # Sum the boolean masks across keywords to get scores per document
    keyword_scores = jnp.sum(jnp.array(keyword_masks), axis=0).astype(jnp.float32)
    return keyword_scores

def optimized_query(num_docs, doc_lengths, avgdl, inverted_index, idf, k1, b,
                    query_terms, texts, metadata, metadata_filter, top_k,
                    variant="bm25", delta=0.5,threshold=0.01):
    """
    Optimized query function for BM25 with variations using JAX.
    
    Parameters:
      - num_docs: number of documents.
      - doc_lengths: jnp.array of shape (num_docs,) with document lengths.
      - avgdl: average document length.
      - inverted_index: dict mapping term -> list of (doc_id, term frequency).
      - idf: dict mapping term -> idf value.
      - k1, b: BM25 parameters.
      - query_terms: list of query terms (strings).
      - texts: list of document texts (strings).
      - metadata: list of document metadata (dicts).
      - metadata_filter: dict for filtering documents by metadata.
      - top_k: number of top results to return.
      - variant: BM25 variant ("bm25", "bm25+", "bm25l", "atire").
      - delta: delta parameter for BM25+ and BM25L.
      
    Returns:
      A list of dicts with document text, metadata, and score.
    """
    idf_values = jnp.array([idf.get(term, 0) for term in query_terms], dtype=jnp.float32)
    scores = jnp.zeros(num_docs, dtype=jnp.float32)
    
    # Compute BM25 scores for each query term
    for term, idf_val in zip(query_terms, idf_values):
        if term in inverted_index:
            term_data = inverted_index[term]  # list of (doc_id, tf)
            term_freqs = jnp.array([tf for doc_id, tf in term_data], dtype=jnp.float32)
            doc_ids = jnp.array([doc_id for doc_id, tf in term_data])
            norm = k1 * (1 - b + b * doc_lengths[doc_ids] / avgdl)
            if variant in ("bm25", "atire"):
                term_scores = idf_val * ((term_freqs * (k1 + 1)) / (term_freqs + norm))
            elif variant == "bm25+":
                term_scores = idf_val * (((term_freqs + delta) * (k1 + 1)) / (term_freqs + norm + delta))
            elif variant == "bm25l":
                term_scores = idf_val * (term_freqs / (term_freqs + norm + delta * (doc_lengths[doc_ids] / avgdl)))
            else:
                term_scores = jnp.zeros_like(term_freqs)
            scores = scores.at[doc_ids].add(term_scores)
    
    # Add keyword match bonus
    keyword_scores = _compute_keyword_scores(texts, query_terms)
    scores = scores + keyword_scores
    
    # Apply metadata filtering if provided
    if metadata_filter:
        mask = jnp.array([
            all(metadata[i].get(key) == val for key, val in metadata_filter.items())
            for i in range(num_docs)
        ], dtype=jnp.float32)
        scores = scores * mask
    
    # Get top_k document indices (descending order)
    top_indices = jnp.argsort(scores,descending=True)[:top_k]
    top_k_scores = scores[top_indices]
    top_k_docs = [{"Texts": texts[int(doc_id)], **metadata[int(doc_id)], "Score": float(score)}
                  for doc_id, score in zip(top_indices, top_k_scores) if score > threshold]
    
    return top_k_docs