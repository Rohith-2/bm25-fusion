from bm25_fusion.core import BM25
from mtbenchmark.retrieval import BaseRetriever  # example interface; adjust if needed
from mtbenchmark.datasets import MTEBDataset  # adjust to your MTEB installation
from tqdm import tqdm

class BM25Retriever(BaseRetriever):
    def __init__(self, texts, metadata, k1=1.5, b=0.75, delta=0.5):
        self.bm25 = BM25(texts=texts, metadata=metadata, k1=k1, b=b, delta=delta)
    
    def add_documents(self, texts, metadata):
        # Assumes texts is a list of document strings.
        # Tokenize as needed before calling add_document
        # Here we assume BM25.tokenize is used internally.
        for text, meta in tqdm(zip(texts, metadata),total=len(texts)):
            self.bm25.add_document([text], [meta])
    
    def retrieve(self, query, top_k=10, metadata_filter=None):
        return self.bm25.query(query, top_k=top_k, metadata_filter=metadata_filter)

def main():
    # Load a dataset from MTEB. Here we pick one (e.g., ArguAna).
    dataset = MTEBDataset("ArguAna")  # change to an available dataset
    # Use the corpus (pass as texts) and metadata from the dataset.
    texts = dataset.corpus  # adjust attribute names as needed
    metadata = dataset.metadata  # adjust to match your BM25 expectations

    # Instantiate retriever
    retriever = BM25Retriever(texts, metadata)
    
    # Benchmark retrieval using the standard evaluation function
    results = dataset.evaluate_retriever(retriever, top_k=10)
    print("Benchmark results:", results)

if __name__ == "__main__":
    main()