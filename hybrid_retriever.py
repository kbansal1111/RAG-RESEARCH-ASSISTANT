import chromadb
import numpy as np
from bm25_retriever import BM25Retriever
from embedding_model import model
from config import Config

class HybridRetriever:
    """Hybrid search combining BM25 and dense embeddings with score fusion"""
    
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name=Config.COLLECTION_NAME)
        self.dense_weight = Config.DENSE_WEIGHT
        self.bm25_weight = Config.BM25_WEIGHT
        
    def index(self, chunks, embeddings):
        """Index documents for both BM25 and dense search"""
        # Index for BM25
        self.bm25_retriever.index(chunks)
        
        # Index for dense search (ChromaDB)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        try:
            self.collection.delete(ids=ids)
        except:
            pass
            
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )
        
    def _normalize_scores(self, scores):
        """Min-max normalization of scores"""
        if len(scores) == 0:
            return scores
        scores = np.array(scores)
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
        
    def search(self, query, k=Config.TOP_K_DENSE):
        """Hybrid search with score fusion"""
        # BM25 search
        bm25_results = self.bm25_retriever.search(query, k=k)
        bm25_scores_normalized = self._normalize_scores(bm25_results['scores'])
        
        # Dense search
        query_embedding = model.encode(query)
        dense_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Get dense scores and normalize
        dense_scores = dense_results['distances'][0]
        dense_scores_normalized = self._normalize_scores([-s for s in dense_scores])  # Convert distance to similarity
        
        # Create score mapping for fusion
        score_map = {}
        
        # Add BM25 scores
        for idx, score in zip(bm25_results['indices'], bm25_scores_normalized):
            chunk_id = f"chunk_{idx}"
            score_map[chunk_id] = score_map.get(chunk_id, 0) + score * self.bm25_weight
            
        # Add dense scores
        for doc, score in zip(dense_results['documents'][0], dense_scores_normalized):
            # Find the chunk_id by matching document content
            for idx, original_doc in enumerate(self.bm25_retriever.corpus):
                if original_doc == doc:
                    chunk_id = f"chunk_{idx}"
                    score_map[chunk_id] = score_map.get(chunk_id, 0) + score * self.dense_weight
                    break
        
        # Sort by fused scores
        sorted_results = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        top_k = min(k, len(sorted_results))
        final_indices = [int(chunk_id.split('_')[1]) for chunk_id, _ in sorted_results[:top_k]]
        final_scores = [score for _, score in sorted_results[:top_k]]
        final_docs = [self.bm25_retriever.corpus[idx] for idx in final_indices]
        
        return {
            'indices': final_indices,
            'scores': final_scores,
            'documents': final_docs
        }
