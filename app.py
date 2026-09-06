from hybrid_retriever import HybridRetriever
from reranker import Reranker
from llm import LLMGenerator
from token_manager import TokenManager
from cache import SemanticCache
from monitor import PerformanceMonitor, monitor_latency, logger
from config import Config
import time

class EnhancedRAGPipeline:
    """FAANG-ready RAG pipeline with hybrid search, reranking, caching, and monitoring"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.llm_generator = LLMGenerator(monitor=self.monitor)
        self.token_manager = TokenManager()
        self.cache = SemanticCache()
        
    @monitor_latency('total_query_time')
    def process_query(self, query):
        """Process a query through the full RAG pipeline"""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(query)
        if cached_result:
            self.monitor.record_cache_hit()
            logger.info("Cache hit - returning cached result")
            return cached_result['answer'], cached_result['context']
        
        self.monitor.record_cache_miss()
        logger.info("Cache miss - processing query")
        
        # Hybrid search (BM25 + dense)
        hybrid_results = self.hybrid_retriever.search(query, k=Config.TOP_K_DENSE)
        
        # Rerank results
        reranked_results = self.reranker.rerank(
            query, 
            hybrid_results['documents'],
            hybrid_results['scores'],
            k=Config.TOP_K_FINAL
        )
        
        # Token budget management
        selected_chunks, total_tokens = self.token_manager.select_chunks_within_budget(
            reranked_results['documents'],
            query
        )
        
        logger.info(f"Selected {len(selected_chunks)} chunks within token budget ({total_tokens} tokens)")
        
        # Build context
        context = "\n\n".join(selected_chunks)
        
        # Generate answer
        answer = self.llm_generator.generate_answer(query, context)
        
        # Cache result
        self.cache.set(query, {
            'answer': answer,
            'context': context
        })
        
        total_time = time.time() - start_time
        logger.info(f"Query processed in {total_time:.2f}s")
        
        return answer, context
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return self.monitor.get_stats()
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return self.cache.get_stats()

def main():
    """Main CLI interface"""
    pipeline = EnhancedRAGPipeline()
    
    print("=" * 60)
    print("FAANG-Ready RAG Research Assistant")
    print("Features: Hybrid Search | Reranking | Caching | Monitoring")
    print("=" * 60)
    
    while True:
        query = input("\nAsk Question (or 'stats' for performance, 'exit' to quit): ")
        
        if query.lower() == "exit":
            break
        elif query.lower() == "stats":
            print("\n--- Performance Statistics ---")
            perf_stats = pipeline.get_performance_stats()
            for stage, stats in perf_stats.items():
                print(f"{stage}: {stats}")
            
            print("\n--- Cache Statistics ---")
            cache_stats = pipeline.get_cache_stats()
            for key, value in cache_stats.items():
                print(f"{key}: {value}")
            continue
        
        try:
            answer, context = pipeline.process_query(query)
            
            print("\n" + "=" * 60)
            print("Answer:")
            print("=" * 60)
            print(answer)
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"\nError: {str(e)}")
    
    print("\nFinal Performance Statistics:")
    print(pipeline.get_performance_stats())
    print("\nCache Statistics:")
    print(pipeline.get_cache_stats())

if __name__ == "__main__":
    main()