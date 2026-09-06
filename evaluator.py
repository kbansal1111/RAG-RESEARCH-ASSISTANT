import numpy as np
from typing import List, Dict

class Evaluator:
    """Evaluation framework for retrieval system metrics"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate precision@k"""
        if k == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_count = sum(1 for doc in retrieved_k if doc in relevant_docs)
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate recall@k"""
        if len(relevant_docs) == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_count = sum(1 for doc in retrieved_k if doc in relevant_docs)
        return relevant_count / len(relevant_docs)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Average Precision"""
        if len(relevant_docs) == 0:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_docs) if relevant_count > 0 else 0.0
    
    @staticmethod
    def evaluate_retrieval(retrieved_docs: List[str], relevant_docs: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Comprehensive retrieval evaluation"""
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = Evaluator.precision_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'recall@{k}'] = Evaluator.recall_at_k(retrieved_docs, relevant_docs, k)
        
        metrics['mrr'] = Evaluator.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        metrics['map'] = Evaluator.average_precision(retrieved_docs, relevant_docs)
        
        return metrics
    
    @staticmethod
    def evaluate_answer_quality(generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Simple answer quality evaluation (can be extended with more sophisticated metrics)"""
        # Basic overlap metrics
        gen_words = set(generated_answer.lower().split())
        ref_words = set(reference_answer.lower().split())
        
        if len(ref_words) == 0:
            return {'jaccard_similarity': 0.0, 'coverage': 0.0}
        
        intersection = gen_words & ref_words
        union = gen_words | ref_words
        
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        coverage = len(intersection) / len(ref_words) if len(ref_words) > 0 else 0.0
        
        return {
            'jaccard_similarity': jaccard,
            'coverage': coverage
        }
