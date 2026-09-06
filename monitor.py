import time
import logging
from typing import Dict, List
from functools import wraps
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring and latency tracking for RAG pipeline"""
    
    def __init__(self):
        self.metrics = {
            'embedding_generation': [],
            'bm25_search': [],
            'dense_search': [],
            'reranking': [],
            'llm_generation': [],
            'total_query_time': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def record_latency(self, stage: str, latency_ms: float):
        """Record latency for a pipeline stage"""
        if stage in self.metrics:
            self.metrics[stage].append(latency_ms)
            
            if Config.ENABLE_LATENCY_TRACKING and latency_ms > Config.LOG_LATENCY_THRESHOLD_MS:
                logger.warning(f"High latency detected in {stage}: {latency_ms:.2f}ms")
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.metrics['cache_hits'] += 1
        
    def record_cache_miss(self):
        """Record a cache miss"""
        self.metrics['cache_misses'] += 1
        
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        
        for stage, latencies in self.metrics.items():
            if isinstance(latencies, list) and len(latencies) > 0:
                stats[stage] = {
                    'mean_ms': np.mean(latencies),
                    'median_ms': np.median(latencies),
                    'min_ms': np.min(latencies),
                    'max_ms': np.max(latencies),
                    'count': len(latencies)
                }
            elif isinstance(latencies, int):
                stats[stage] = latencies
                
        # Calculate cache hit rate
        total_cache_ops = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_ops > 0:
            stats['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache_ops
            
        return stats
    
    def reset(self):
        """Reset all metrics"""
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            else:
                self.metrics[key] = 0

def monitor_latency(stage_name: str):
    """Decorator to monitor function latency"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = args[0].monitor if hasattr(args[0], 'monitor') else None
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            if monitor:
                monitor.record_latency(stage_name, latency_ms)
                logger.info(f"{stage_name} completed in {latency_ms:.2f}ms")
            
            return result
        return wrapper
    return decorator

# Import numpy for stats
import numpy as np
