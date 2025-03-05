# BM25 Fusion

BM25-Fusion is an ultra‑fast, flexible BM25 retriever with Meta-Data Filtering  & Real Time modifiable library written in Python and optimised via Numba. It implements several variants of the BM25 algorithm—including classic BM25 (Lucene/Robertson), BM25+, BM25L, and ATIRE—with support for eager indexing, metadata filtering, and stopword removal. By fusing these capabilities into one modular package, BM25 Fusion delivers efficient and scalable retrieval performance suitable for large datasets and modern search applications.

## Features

- **Metadata Filtering:**  
  Supports filtering of results based on metadata (e.g., category, author) by applying a mask on the computed scores.

- **Multiple BM25 Variants:**  
  Supports classic BM25 (Lucene/Robertson), BM25+, BM25L, and ATIRE variants. Configure the variant and delta parameter to adjust scoring as needed.

- **Real Time Addition & Deletion**
  Supports adding and removing docs efficienty.

- **Eager Indexing:**  
  Precomputes BM25 score contributions for each document token and stores them in a sparse matrix. This minimizes query-time computations.

- **Stopword Removal:**  
  Integrates stopword removal during both indexing and query processing to improve retrieval quality.

- **Modular Design:**  
  Clean, modular codebase divided into core BM25 scoring, retrieval functions, and tokenization—making it easy to maintain and extend.

- **High Performance:**  
  Uses Numba‑accelerated functions for parallelized score computations and leverages SciPy’s sparse matrices for memory efficiency and speed.

## Installation

You can install BM25 Fusion directly from source or from PyPI (once released). To install from source:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Rohith-2/bm25_fusion.git
   cd bm25_fusion
   ```

2. **Install in editable mode:**

   ```bash
   pip install -e .
   ```

Alternatively, if the package is available on PyPI:

```bash
pip install bm25_fusion
```

## Quick Start

Below is an example demonstrating how to build a BM25 index and run a query:

```python
from bm25_fusion import BM25
from bm25_fusion.tokenization import tokenize

# Sample corpus and metadata
corpus = [
    "hello world",
    "machine learning is fun",
    "hello machine"
]
metadata = [
    {"category": "news", "author": "Alice"},
    {"category": "science", "author": "Bob"},
    {"category": "news", "author": "Charlie"}
]

# Create BM25 index using the BM25+ variant with stopword removal
stopwords = {"is", "a", "the", "and"}
bm25 = BM25(
    metadata=metadata,
    texts=corpus,
    variant="bm25+",
    delta=0.5,
    stopwords=stopwords
)

bm25.add_document(['This is the hello document with jax in it'], [{'author': 'Me', 'year': 2021}])


# Run a query with metadata filtering (e.g., only "science" documents)
query = "machine learning"
results = bm25.query(query, metadata_filter={"category": "science"}, top_k=2)

for res in results:
    print(f"Score: {res['score']:.4f} | Text: {res['text']} | Metadata: {res}")
```

## Supported BM25 Variants

BM25 Fusion allows you to switch between different scoring models by specifying the `variant` parameter when creating an instance:

- **"bm25" / "lucene" / "robertson":** Classic BM25 scoring.
- **"bm25+":** Adds a delta parameter to ensure non‑zero contributions from rare terms.
- **"bm25l":** Applies a linear normalization adjustment for document length.
- **"atire":** Similar to classic BM25, but clamps the inverse document frequency (idf) to non‑negative values.

You can adjust the `delta` parameter as needed (default is 0.5).

## API Reference

### BM25 Class

**Constructor:**

```python
BM25(metadata=None, texts=None, k1=1.5, b=0.75,
     variant="bm25", delta=0.5, stopwords=None)
```

- `metadata`: A list of dictionaries containing document metadata.
- `texts`: A list of raw text strings corresponding to each document.
- `k1`: BM25 k1 parameter.
- `b`: BM25 b parameter.
- `variant`: BM25 variant to use ("bm25", "bm25+", "bm25l", or "atire").
- `delta`: Delta parameter (used in BM25+ and BM25L).
- `stopwords`: An iterable of stopwords to remove from documents and queries.

**Methods:**

- `query(query_tokens, metadata_filter=None, top_k=10)`:  
  Runs a query against the indexed corpus.
  - `query_tokens`: The query in string format or list[str]
  - `metadata_filter`: A dictionary specifying metadata constraints (e.g., `{"category": "science"}`).
  - `top_k`: The number of top results to return.
  
- `save(path)`:  
  Saves the BM25 index to a file using pickle.

- `load(path)`:  
  Loads a BM25 index from a file.

## Tokenization

The package includes a simple tokenization function in `bm25_fusion/tokenization.py`. The default tokenizer splits text on whitespace and removes punctuation. You can replace or extend it as needed.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests with improvements or bug fixes. When contributing:
- Follow the existing coding style.
- Write unit tests for new features.
- Update documentation as needed.

To-Do:
- Benchmark with Standerd Retrieval Datasets.

## License

BM25 Fusion is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Inspired by [bm25s](https://github.com/xhluca/bm25s) and related BM25 implementations.
- Thanks to the communities behind BM25 research.
