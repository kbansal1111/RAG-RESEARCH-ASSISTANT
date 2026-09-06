from sentence_transformers import CrossEncoder
import numpy as np
from config import Config

class Reranker:
    """Cross-encoder reranking for improved retrieval accuracy"""
    
    def __init__(self):
        self.model = None
        if Config.ENABLE_RERANKING:
            print(f"Loading cross-encoder model: {Config.RERANK_MODEL}")
            self.model = CrossEncoder(Config.RERANK_MODEL)
            
    def rerank(self, query, documents, scores=None, k=Config.TOP_K_FINAL):
        """Rerank documents using cross-encoder"""
        if self.model is None:
            # If reranking disabled, return top-k by original scores
            if scores is not None:
                sorted_indices = np.argsort(scores)[::-1][:k]
                return {
                    'documents': [documents[i] for i in sorted_indices],
                    'scores': [scores[i] for i in sorted_indices]
                }
            return {
                'documents': documents[:k],
                'scores': [1.0] * min(k, len(documents))
            }
        
        # Prepare query-document pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Get cross-encoder scores
        rerank_scores = self.model.predict(pairs)
        
        # Sort by rerank scores
        sorted_indices = np.argsort(rerank_scores)[::-1][:k]
        
        return {
            'documents': [documents[i] for i in sorted_indices],
            'scores': [float(rerank_scores[i]) for i in sorted_indices]
        }
