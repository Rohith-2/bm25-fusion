from bm25_fusion.core import BM25

def main():
        
    # Sample texts (original documents in string format)
    texts = [
        "Hello world, this is a document.",
        "Machine learning is fun with JAX.",
        "This is another document with JAX.",
        "Hello, this is JAX optimization."
    ]
    
    # Example metadata (let's assume we have some metadata)
    metadata = [
        {"author": "Alice", "year": 2021},
        {"author": "Bob", "year": 2022},
        {"author": "Charlie", "year": 2021},
        {"author": "Alice", "year": 2023},
    ]
    
    # Create an instance of BM25Optimized
    bm25 = BM25(texts=texts,metadata=metadata, k1=1.5, b=0.75, delta=0.5)
    bm25.add_document(['This is the hello document with jax in it'], [{'author': 'Alice', 'year': 2021}])
    
    # Define query tokens (tokenized search query)
    query = "hello jax"
    
    # Define metadata filter (e.g., find documents written by Alice)
    metadata_filter = {"author": "Alice"}
    
    # Perform query with keyword match bonus and metadata filtering
    top_k_results = bm25.query(query, top_k=2, metadata_filter=metadata_filter)
    
    # Display the top-k results
    print("Top K results:")
    for doc in top_k_results:
        print(doc, "\n")

# Call the main function to run the example
if __name__ == "__main__":
    main()