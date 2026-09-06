from rank_bmfs import BM25Okapi
import numpy as np
from config import Config

class BM25Retriever:
    """BM25 keyword-based retrieval for hybrid search"""
    
    def __init__(self):
        self.bm25 = None
        self.corpus = None
        self.tokenized_corpus = None
        
    def index(self, chunks):
        """Index documents for BM25 retrieval"""
        self.corpus = chunks
        # Tokenize corpus (simple whitespace tokenization)
        self.tokenized_corpus = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def search(self, query, k=Config.TOP_K_BM25):
        """Search using BM25 and return scores with indices"""
        if self.bm25 is None:
            raise ValueError("BM25 not indexed. Call index() first.")
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]
        
        return {
            'indices': top_indices.tolist(),
            'scores': top_scores.tolist(),
            'documents': [self.corpus[i] for i in top_indices]
        }
