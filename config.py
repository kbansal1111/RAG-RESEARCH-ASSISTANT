import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration management for RAG pipeline"""
    
    # API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "qwen/qwen3-235b-a22b"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Chunking Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # Retrieval Configuration
    TOP_K_DENSE = 10  # Retrieve more for reranking
    TOP_K_FINAL = 3   # Final results after reranking
    TOP_K_BM25 = 10
    
    # Hybrid Search Weights
    DENSE_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # Token Budget Configuration
    MAX_CONTEXT_TOKENS = 3000
    TOKEN_MODEL = "cl100k_base"  # tiktoken encoding
    
    # Caching Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    CACHE_TTL = 3600  # 1 hour
    
    # Performance Monitoring
    ENABLE_LATENCY_TRACKING = True
    LOG_LATENCY_THRESHOLD_MS = 1000
    
    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    RETRY_BACKOFF = 2  # exponential backoff multiplier
    
    # Cross-Encoder Configuration
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_RERANKING = True
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "research_docs"
    
    # Data Configuration
    DATA_DIR = "./data"
