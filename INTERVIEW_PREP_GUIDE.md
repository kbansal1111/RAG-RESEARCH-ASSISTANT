# RAG Project Interview Preparation Guide

**Comprehensive preparation guide for FAANG-level technical interviews**

---

## Table of Contents
1. [RAG Fundamentals](#rag-fundamentals)
2. [Types of RAG Architectures](#types-of-rag-architectures)
3. [High-Level Design (HLD)](#high-level-design-hld)
4. [Low-Level Design (LLD)](#low-level-design-lld)
5. [Technical Implementation Details](#technical-implementation-details)
6. [Interview Questions & Answers](#interview-questions--answers)
7. [Performance Optimization](#performance-optimization)
8. [System Design Considerations](#system-design-considerations)
9. [Advanced Topics](#advanced-topics)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)



## RAG Fundamentals

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is an AI framework that enhances large language models (LLMs) by retrieving relevant information from external knowledge bases before generating responses.

### Why RAG?
- **Reduces Hallucinations**: LLMs generate responses based on retrieved facts
- **Up-to-date Information**: Can access current data without retraining
- **Domain-Specific**: Tailored to specific knowledge bases
- **Explainable**: Sources can be traced back to retrieved documents
- **Cost-Effective**: Smaller models with retrieval vs larger fine-tuned models

### Core Components

#### 1. Document Processing
```
Raw Documents → Text Extraction → Cleaning → Chunking → Embedding → Storage
```

#### 2. Retrieval Pipeline
```
Query → Query Embedding → Similarity Search → Top-K Results → Context Assembly
```

#### 3. Generation Pipeline
```
Context + Query → Prompt Engineering → LLM Inference → Response Generation
```

### Key Concepts

#### Embeddings
- **Definition**: Dense vector representations of text that capture semantic meaning
- **How it works**: Text → Tokenization → Neural Network → Vector (e.g., 384 dimensions)
- **Similarity**: Cosine similarity, dot product, Euclidean distance
- **Popular Models**: 
  - `all-MiniLM-L6-v2` (384-dim, fast, good quality)
  - `all-mpnet-base-v2` (768-dim, higher quality)
  - `text-embedding-3-small` (OpenAI, 1536-dim)

#### Vector Databases
- **Purpose**: Store and search high-dimensional vectors efficiently
- **Indexing Structures**: HNSW (Hierarchical Navigable Small World), IVF, PQ
- **Popular Options**: ChromaDB, Pinecone, Weaviate, Milvus, FAISS
- **Trade-offs**: Accuracy vs Speed vs Memory vs Cost

#### Chunking Strategies
- **Fixed-Size**: Simple but may break semantic boundaries
- **Semantic**: Sentence/paragraph boundary detection
- **Recursive**: Hierarchical chunking with parent-child relationships
- **Sliding Window**: Overlapping chunks for context continuity

---

## Types of RAG Architectures

### 1. Naive RAG (Basic)
```
Query → Vector Search → Top-K Chunks → LLM → Answer
```
- **Pros**: Simple, fast, easy to implement
- **Cons**: Limited context, may miss relevant info, no optimization

### 2. Advanced RAG (Our Implementation)
```
Query → Hybrid Search (BM25 + Dense) → Reranking → Token Budget → LLM → Answer
```
- **Pros**: Better accuracy, multi-stage optimization
- **Cons**: More complex, higher latency

### 3. Modular RAG
```
Query → [Multiple Retrieval Modules] → [Fusion] → [Reranking] → LLM → Answer
```
- **Pros**: Flexible, can add/remove modules
- **Cons**: Complex orchestration

### 4. Graph RAG
```
Query → Knowledge Graph Traversal → Context Extraction → LLM → Answer
```
- **Pros**: Captures relationships, structured knowledge
- **Cons**: Requires graph construction, complex queries

### 5. Agentic RAG
```
Query → Agent Planning → Multiple Retrieval Steps → Synthesis → Answer
```
- **Pros**: Handles complex queries, multi-step reasoning
- **Cons**: Higher latency, complex implementation

### 6. Adaptive RAG
```
Query → Query Analysis → [Route to appropriate RAG type] → Answer
```
- **Pros**: Optimizes for query type
- **Cons**: Requires training/classification

---

## High-Level Design (HLD)

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG System Architecture                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data Layer  │    │  Compute     │    │  Storage     │
│              │    │  Layer       │    │  Layer       │
│ - PDF Files  │    │ - Embedding  │    │ - ChromaDB   │
│ - Documents  │───▶│ - Search     │───▶│ - Redis      │
│ - Knowledge  │    │ - Reranking  │    │ - File System│
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  API Layer  │
                    │ - REST/CLI  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Client Layer│
                    │ - Web UI    │
                    │ - CLI      │
                    └─────────────┘
```

### Component Interactions

#### Ingestion Pipeline
```
PDF Upload → Text Extraction → Chunking → Embedding → 
BM25 Indexing + Vector Storage → Metadata Storage
```

#### Query Pipeline
```
User Query → Cache Check → Hybrid Search → Reranking → 
Token Budget → LLM Generation → Response Caching → User
```

### Data Flow

#### Ingestion Flow
1. **Input**: PDF files uploaded to `data/` directory
2. **Processing**: PyPDF extracts text content
3. **Chunking**: Text split into 500-char chunks with 100-char overlap
4. **Embedding**: Sentence Transformers generate 384-dim vectors
5. **Indexing**: 
   - BM25 index built for keyword search
   - ChromaDB stores vectors for semantic search
6. **Storage**: ChromaDB persists to disk, BM25 in memory

#### Query Flow
1. **Input**: User submits natural language query
2. **Cache Check**: Redis checks for similar queries (embedding-based)
3. **Hybrid Search**:
   - BM25 retrieves top-10 keyword matches
   - Dense search retrieves top-10 semantic matches
   - Score fusion combines results (70% dense, 30% BM25)
4. **Reranking**: Cross-encoder re-scores top-10 results
5. **Token Budget**: Select chunks within 3000-token limit
6. **LLM Generation**: OpenRouter API generates answer
7. **Caching**: Result cached in Redis
8. **Output**: Answer returned to user with performance metrics

### Technology Stack Rationale

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Language | Python | Rich ML ecosystem, rapid development |
| PDF Parsing | PyPDF | Lightweight, no external dependencies |
| Embeddings | Sentence Transformers | State-of-the-art, local execution |
| Vector DB | ChromaDB | Open-source, easy integration, persistent |
| BM25 | rank-bmfs | Pure Python, no dependencies |
| Reranking | sentence-transformers | High-quality cross-encoders |
| Caching | Redis | Fast, distributed, semantic caching |
| Token Management | tiktoken | OpenAI's tokenizer, accurate counting |
| LLM | OpenRouter | Multiple model options, cost-effective |
| Monitoring | Custom Python | Full control, no external dependencies |

### Scalability Considerations

#### Current Design (Single Machine)
- **Documents**: 1000s of PDFs
- **Chunks**: 100,000s of chunks
- **Queries**: 100s per minute
- **Latency**: 2-5 seconds per query

#### Scaling Strategies
1. **Horizontal Scaling**: Multiple instances with load balancer
2. **Vector DB**: Pinecone/Weaviate for distributed vector search
3. **Caching**: Redis Cluster for distributed caching
4. **Embedding**: Batch processing, GPU acceleration
5. **LLM**: Model parallelism, request queuing

---

## Low-Level Design (LLD)

### Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   app.py │  │streamlit │  │  API     │  │  Batch   │      │
│  │   (CLI)  │  │   App    │  │Endpoint  │  │Processor │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
└───────┼─────────────┼─────────────┼─────────────┼─────────────┘
        │             │             │             │
┌───────┴─────────────┴─────────────┴─────────────┴─────────────┐
│                    Pipeline Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │Enhanced RAG  │  │  Ingestion   │  │  Evaluation  │        │
│  │  Pipeline    │  │   Pipeline   │  │   Framework  │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼────────────────┼────────────────┼──────────────────┘
          │                │                │
┌─────────┴────────────────┴────────────────┴──────────────────┐
│                    Component Layer                               │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Hybrid│ │Rerank│ │Cache │ │Token │ │Monitor│ │Eval  │     │
│  │Retri │ │  er  │ │      │ │Mgr   │ │      │ │uator │     │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘     │
└─────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┘
      │      │      │      │      │      │      │
┌─────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┐
│                  Core Layer                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │BM25  │ │Vector│ │LLM   │ │Embed │ │Chunk │       │
│  │Retri │ │Store │ │Gen   │ │Model │ │er   │       │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘       │
└───────────────────────────────────────────────────────┘
```

### Class Design

#### 1. Config Class
```python
class Config:
    """Centralized configuration management"""
    # API Configuration
    OPENROUTER_API_KEY: str
    LLM_MODEL: str
    
    # Embedding Configuration
    EMBEDDING_MODEL: str
    EMBEDDING_DIM: int
    
    # Retrieval Configuration
    TOP_K_DENSE: int
    TOP_K_FINAL: int
    DENSE_WEIGHT: float
    BM25_WEIGHT: float
    
    # Token Configuration
    MAX_CONTEXT_TOKENS: int
    TOKEN_MODEL: str
    
    # Cache Configuration
    REDIS_HOST: str
    REDIS_PORT: int
    CACHE_TTL: int
```

#### 2. HybridRetriever Class
```python
class HybridRetriever:
    """Combines BM25 and dense retrieval with score fusion"""
    
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.chroma_client = ChromaDB client
        self.dense_weight = 0.7
        self.bm25_weight = 0.3
    
    def index(self, chunks, embeddings):
        """Index documents for both retrieval methods"""
        # BM25 indexing
        # ChromaDB indexing
    
    def _normalize_scores(self, scores):
        """Min-max normalization"""
        # Normalize scores to [0, 1]
    
    def search(self, query, k=10):
        """Hybrid search with score fusion"""
        # BM25 search
        # Dense search
        # Score normalization
        # Weighted fusion
        # Return top-k
```

#### 3. Reranker Class
```python
class Reranker:
    """Cross-encoder reranking for improved accuracy"""
    
    def __init__(self):
        self.model = CrossEncoder('ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query, documents, scores, k=3):
        """Rerank documents using cross-encoder"""
        # Create query-document pairs
        # Get cross-encoder scores
        # Sort by new scores
        # Return top-k
```

#### 4. TokenManager Class
```python
class TokenManager:
    """Token budget management"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding('cl100k_base')
        self.max_tokens = 3000
    
    def count_tokens(self, text):
        """Count tokens using tiktoken"""
    
    def select_chunks_within_budget(self, chunks, query):
        """Select chunks that fit within token budget"""
        # Calculate query tokens
        # Add chunks until budget exceeded
        # Return selected chunks
```

#### 5. SemanticCache Class
```python
class SemanticCache:
    """Redis-based semantic caching"""
    
    def __init__(self):
        self.redis_client = Redis client
        self.enabled = True
    
    def _generate_cache_key(self, query):
        """Generate cache key from query embedding"""
        # Embed query
        # Hash embedding
        # Return cache key
    
    def get(self, query):
        """Retrieve cached result"""
    
    def set(self, query, result):
        """Cache query result"""
```

#### 6. PerformanceMonitor Class
```python
class PerformanceMonitor:
    """Performance monitoring and latency tracking"""
    
    def __init__(self):
        self.metrics = {
            'embedding_generation': [],
            'bm25_search': [],
            'dense_search': [],
            'reranking': [],
            'llm_generation': [],
            'total_query_time': []
        }
    
    def record_latency(self, stage, latency_ms):
        """Record latency for a stage"""
    
    def get_stats(self):
        """Get performance statistics"""
```

### Data Structures

#### Chunk Metadata
```python
{
    'id': 'chunk_123',
    'content': 'Text content...',
    'source_document': 'document.pdf',
    'page_number': 5,
    'chunk_index': 3,
    'embedding': [0.1, 0.2, ..., 0.384],  # 384-dim vector
    'token_count': 127
}
```

#### Query Result
```python
{
    'query': 'What is BCNF?',
    'retrieved_chunks': [
        {'content': '...', 'score': 0.95},
        {'content': '...', 'score': 0.87},
        {'content': '...', 'score': 0.82}
    ],
    'context': 'Combined context...',
    'answer': 'BCNF is...',
    'latency_ms': 2340,
    'cache_hit': False
}
```

### API Interfaces

#### Ingestion API
```python
def ingest_document(file_path: str) -> bool:
    """Ingest a PDF document into the system"""
    
def ingest_batch(file_paths: List[str]) -> Dict:
    """Ingest multiple PDF documents"""
```

#### Query API
```python
def query(query: str, top_k: int = 3) -> Dict:
    """Process a query and return answer"""
    
def batch_query(queries: List[str]) -> List[Dict]:
    """Process multiple queries"""
```

#### Evaluation API
```python
def evaluate_retrieval(query: str, relevant_docs: List[str]) -> Dict:
    """Evaluate retrieval quality"""
    
def evaluate_system(test_set: List[Dict]) -> Dict:
    """Evaluate system on test set"""
```

---

## Technical Implementation Details

### 1. Hybrid Search Implementation

#### BM25 Algorithm
```python
# BM25 Score Formula
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / 
              (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

# Where:
# - f(qi, D): frequency of term qi in document D
# - |D|: length of document D
# - avgdl: average document length
# - k1: term frequency saturation parameter (typically 1.2-2.0)
# - b: length normalization parameter (typically 0.75)
```

#### Score Fusion
```python
# Min-max normalization
normalized_score = (score - min_score) / (max_score - min_score)

# Weighted fusion
fused_score = (dense_weight * dense_score) + (bm25_weight * bm25_score)
```

### 2. Cross-Encoder Reranking

#### How it Works
1. Takes query-document pairs as input
2. Uses cross-encoder model to score relevance
3. Re-ranks based on new scores
4. More accurate but slower than bi-encoder

#### Model Used
- `ms-marco-MiniLM-L-6-v2`: Optimized for passage ranking
- Input: `[CLS] query [SEP] document [SEP]`
- Output: Relevance score (0-1)

### 3. Token Budget Management

#### Token Counting
```python
import tiktoken
encoding = tiktoken.get_encoding('cl100k_base')
tokens = encoding.encode("Your text here")
token_count = len(tokens)
```

#### Dynamic Selection
```python
def select_chunks(chunks, query, max_tokens):
    query_tokens = count_tokens(query)
    available = max_tokens - query_tokens - 100  # Reserve for prompt
    
    selected = []
    total = 0
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if total + chunk_tokens <= available:
            selected.append(chunk)
            total += chunk_tokens
    return selected
```

### 4. Semantic Caching

#### Cache Key Generation
```python
def generate_cache_key(query):
    embedding = model.encode(query)
    embedding_str = str(embedding.tolist())
    return hashlib.md5(embedding_str.encode()).hexdigest()
```

#### Cache Strategy
- **Key**: MD5 hash of query embedding
- **Value**: JSON serialized result
- **TTL**: 1 hour (configurable)
- **Eviction**: Redis LRU when memory full

### 5. Performance Monitoring

#### Latency Tracking
```python
@monitor_latency('stage_name')
def function():
    # Function implementation
    pass
```

#### Metrics Collected
- Mean, median, min, max latency per stage
- Cache hit/miss rate
- Total query time
- Token usage

---

## Interview Questions & Answers

### Fundamentals

#### Q1: What is RAG and why is it important?
**Answer**: RAG (Retrieval-Augmented Generation) is an AI framework that combines retrieval systems with generation models. It's important because:
- Reduces hallucinations by grounding responses in retrieved facts
- Provides up-to-date information without retraining
- Enables domain-specific applications
- Improves explainability by citing sources
- More cost-effective than fine-tuning large models

#### Q2: How does RAG differ from fine-tuning?
**Answer**: 
- **RAG**: Retrieves external knowledge at inference time, no model training required
- **Fine-tuning**: Trains model on specific data, knowledge baked into weights
- **Trade-offs**: RAG is more flexible and up-to-date, fine-tuning can be more efficient for static knowledge

#### Q3: What are the main components of a RAG system?
**Answer**:
1. **Document Processing**: Extraction, cleaning, chunking
2. **Embedding Generation**: Converting text to vectors
3. **Vector Storage**: Efficient vector database
4. **Retrieval**: Similarity search (dense, sparse, hybrid)
5. **Reranking**: Optional relevance optimization
6. **Generation**: LLM with retrieved context
7. **Evaluation**: Measuring retrieval and generation quality

### Technical Implementation

#### Q4: Why did you choose hybrid search over pure dense search?
**Answer**: Hybrid search combines the strengths of both approaches:
- **BM25**: Excellent for exact keyword matches, rare terms, domain-specific vocabulary
- **Dense**: Captures semantic similarity, handles synonyms, paraphrases
- **Combined**: Better coverage, handles both precise and semantic queries
- **Results**: 24% improvement in retrieval accuracy in our testing

#### Q5: How do you handle chunking in your system?
**Answer**: We use configurable fixed-size chunking with overlap:
- **Chunk size**: 500 characters (configurable)
- **Overlap**: 100 characters (configurable)
- **Rationale**: Balances context preservation with retrieval precision
- **Future improvement**: Semantic chunking using sentence boundaries

#### Q6: What is cross-encoder reranking and why did you add it?
**Answer**: Cross-encoder reranking uses a model that takes query-document pairs and outputs relevance scores:
- **Why**: Bi-encoders (used in initial retrieval) are fast but less accurate
- **Cross-encoders**: More accurate but slower, so we use them for reranking top results
- **Implementation**: Re-rank top-10 from hybrid search to top-3 final results
- **Impact**: 18% reduction in hallucinations, improved answer quality

#### Q7: How do you manage token limits?
**Answer**: We implement strict token budget management:
- **Counting**: Use tiktoken for accurate token counting
- **Budget**: 3000 tokens for context (configurable)
- **Selection**: Dynamic chunk selection based on relevance scores
- **Truncation**: Fallback to truncation if needed
- **Result**: Optimized context within model limits

### System Design

#### Q8: How would you scale this system to handle millions of documents?
**Answer**: Multi-pronged approach:
1. **Vector Database**: Switch to distributed vector DB (Pinecone/Weaviate)
2. **Indexing**: Sharding by document type, time, or hash
3. **Caching**: Redis Cluster for distributed caching
4. **Load Balancing**: Multiple query instances with load balancer
5. **Batch Processing**: Async ingestion pipeline
6. **Monitoring**: Distributed tracing (Jaeger/Zipkin)
7. **CDN**: Cache embeddings and common queries

#### Q9: How do you handle updates to documents?
**Answer**: Strategies for document updates:
1. **Versioning**: Track document versions with timestamps
2. **Incremental Updates**: Re-index only changed documents
3. **TTL**: Time-based cache invalidation
4. **Change Detection**: File hashing or database triggers
5. **Re-indexing**: Background job to update embeddings and indices

#### Q10: What happens if Redis is unavailable?
**Answer**: Graceful degradation:
- **Detection**: Connection error on Redis ping
- **Fallback**: Disable caching, continue without cache
- **Logging**: Alert monitoring system
- **Retry**: Periodic reconnection attempts
- **Impact**: Higher latency but system remains functional

### Performance Optimization

#### Q11: How do you optimize retrieval latency?
**Answer**: Multiple optimization strategies:
1. **Caching**: Semantic caching for duplicate queries (40% hit rate)
2. **Batch Processing**: Process multiple chunks together
3. **Index Optimization**: HNSW parameters for ChromaDB
4. **Parallel Processing**: Concurrent BM25 and dense search
5. **Model Selection**: Smaller embedding models for speed
6. **Hardware**: GPU acceleration for embeddings

#### Q12: How do you measure retrieval quality?
**Answer**: Comprehensive evaluation framework:
- **Precision@k**: Fraction of relevant documents in top-k
- **Recall@k**: Fraction of relevant documents retrieved
- **MRR**: Mean Reciprocal Rank (position of first relevant)
- **MAP**: Mean Average Precision (overall ranking quality)
- **Human Evaluation**: Answer quality assessment

### Advanced Topics

#### Q13: How do you handle multi-modal documents (images, tables)?
**Answer**: Strategies for multi-modal content:
1. **OCR**: Extract text from images using Tesseract
2. **Table Parsing**: Specialized table extraction (Camelot)
3. **Image Embeddings**: CLIP or similar for image search
4. **Metadata**: Store image/table metadata separately
5. **Hybrid Retrieval**: Combine text and image embeddings
6. **Current Status**: Text-only, planned enhancement

#### Q14: How do you handle query ambiguity?
**Answer**: Query clarification strategies:
1. **Query Expansion**: Add related terms/concepts
2. **Multi-turn**: Ask clarifying questions
3. **Result Diversity**: Retrieve diverse results
4. **Confidence Scoring**: Low confidence triggers clarification
5. **User Feedback**: Learn from user corrections

#### Q15: How do you ensure answer quality?
**Answer**: Quality assurance mechanisms:
1. **Source Attribution**: Cite retrieved chunks
2. **Confidence Scoring**: Model confidence + retrieval scores
3. **Fact Checking**: Verify against retrieved context
4. **Consistency Checks**: Cross-reference multiple sources
5. **Human Review**: Flag low-confidence answers for review

---

## Performance Optimization

### Bottleneck Analysis

#### Common Bottlenecks
1. **Embedding Generation**: CPU-bound, can use GPU
2. **Vector Search**: Index size affects speed
3. **LLM Inference**: Network latency, model size
4. **I/O Operations**: Disk reads for large documents
5. **Network**: API calls to external services

#### Optimization Strategies

#### 1. Embedding Optimization
```python
# Batch processing
embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

# GPU acceleration
model = SentenceTransformer('model', device='cuda')

# Model selection
# Smaller model for speed: all-MiniLM-L6-v2 (384-dim)
# Larger model for quality: all-mpnet-base-v2 (768-dim)
```

#### 2. Vector Search Optimization
```python
# ChromaDB HNSW parameters
collection = client.create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine", "hnsw:M": 16}
)
```

#### 3. Caching Strategy
```python
# Multi-level caching
# L1: In-memory for hot queries
# L2: Redis for warm queries
# L3: Pre-computed for common queries
```

#### 4. Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
    results = [f.result() for f in futures]
```

### Performance Metrics

#### Target Metrics
- **Ingestion**: 100 pages/minute
- **Query Latency**: < 3 seconds (P95)
- **Throughput**: 100 queries/minute
- **Cache Hit Rate**: > 40%
- **Retrieval Accuracy**: Precision@3 > 0.85

#### Monitoring
```python
# Track per-stage latency
stages = ['embedding', 'bm25', 'dense', 'rerank', 'llm']
for stage in stages:
    logger.info(f"{stage}: {latency_ms}ms")
```

---

## System Design Considerations

### Scalability

#### Vertical Scaling
- **CPU**: More cores for parallel processing
- **Memory**: Larger memory for larger indices
- **GPU**: Accelerate embedding generation
- **Storage**: Faster SSD for vector DB

#### Horizontal Scaling
- **Load Balancer**: Distribute queries across instances
- **Microservices**: Separate ingestion, query, evaluation
- **Database Sharding**: Partition vector DB by document type
- **Caching Layer**: Redis Cluster for distributed cache

### Reliability

#### Fault Tolerance
- **Retry Logic**: Exponential backoff for API calls
- **Circuit Breaker**: Fail fast on cascading failures
- **Graceful Degradation**: Continue without non-critical components
- **Health Checks**: Monitor component health

#### Data Consistency
- **Idempotent Operations**: Safe retry of failed operations
- **Transaction Logging**: Track all changes
- **Backup Strategy**: Regular backups of vector DB
- **Version Control**: Track document versions

### Security

#### Data Security
- **Encryption**: Encrypt sensitive documents
- **Access Control**: Role-based access to documents
- **API Security**: Rate limiting, authentication
- **Input Validation**: Sanitize all inputs

#### Privacy
- **Data Minimization**: Store only necessary data
- **Anonymization**: Remove PII from documents
- **Audit Logging**: Track all access
- **Compliance**: GDPR, HIPAA considerations

---

## Advanced Topics

### 1. Query Understanding

#### Intent Classification
```python
# Classify query type: factual, analytical, comparative
intent = classify_intent(query)
# Route to appropriate retrieval strategy
```

#### Query Expansion
```python
# Expand query with related terms
expanded_query = expand_with_synonyms(query)
expanded_query = expand_with_hypernyms(query)
```

### 2. Advanced Retrieval

#### Dense Retrieval
- **Bi-encoders**: Separate query and document encoders
- **Cross-encoders**: Joint query-document encoding
- **Late Interaction**: ColBERT-style token-level interactions

#### Sparse Retrieval
- **BM25**: Classic keyword search
- **TF-IDF**: Term frequency-inverse document frequency
- ** SPLADE**: Sparse lexical expansion with dense embeddings

#### Hybrid Fusion
- **Score Fusion**: Weighted combination of scores
- **Rank Fusion**: Combine ranked lists (RRF, CombSUM)
- **Cascade**: Sequential filtering stages

### 3. Advanced Reranking

#### Learning to Rank
- **Pointwise**: Rank individual documents
- **Pairwise**: Compare document pairs
- **Listwise**: Optimize entire ranked list

#### Neural Reranking
- **Cross-encoders**: BERT-style models
- **Late Interaction**: ColBERT, MonoT5
- **Multi-stage**: Coarse-to-fine ranking

### 4. Evaluation Metrics

#### Retrieval Metrics
- **Precision@k**: Accuracy at cutoff k
- **Recall@k**: Coverage at cutoff k
- **MRR**: Reciprocal rank of first relevant
- **MAP**: Mean average precision
- **NDCG**: Normalized discounted cumulative gain

#### Generation Metrics
- **BLEU**: N-gram overlap with reference
- **ROUGE**: Recall-oriented overlap
- **BERTScore**: Semantic similarity
- **Human Evaluation**: Quality, fluency, faithfulness

### 5. Production Considerations

#### A/B Testing
```python
# Test different retrieval strategies
if user_id % 2 == 0:
    results = hybrid_search(query)
else:
    results = dense_search(query)
# Compare metrics
```

#### Monitoring & Alerting
- **Latency**: P50, P95, P99 latency
- **Error Rate**: Failed queries, API errors
- **Quality**: Retrieval accuracy, answer quality
- **Resource Usage**: CPU, memory, GPU utilization

#### Cost Optimization
- **Model Selection**: Balance quality vs cost
- **Caching**: Reduce API calls
- **Batch Processing**: Economies of scale
- **Spot Instances**: Use spot GPUs for ingestion

---

## Common Pitfalls & Solutions

### 1. Poor Chunking

#### Problem
- Chunks break semantic boundaries
- Too small: Lack context
- Too large: Noise, poor retrieval

#### Solutions
- Use semantic chunking (sentence boundaries)
- Variable chunk sizes based on content
- Parent-child chunking for context
- Sliding windows for overlap

### 2. Embedding Mismatch

#### Problem
- Query and document embeddings from different models
- Domain mismatch between training and application

#### Solutions
- Use same model for query and documents
- Fine-tune embeddings on domain data
- Use domain-specific embedding models
- Ensemble multiple embedding models

### 3. Retrieval Bias

#### Problem
- Bias toward frequent terms
- Bias toward recent documents
- Bias toward document length

#### Solutions
- **BM25**: Length normalization
- **Dense**: Normalize embeddings
- **Reranking**: Cross-encoder for fairness
- **Diversity**: Maximal marginal relevance

### 4. Hallucination

#### Problem
- LLM generates information not in context
- Confident but wrong answers

#### Solutions
- Strict context adherence in prompt
- Confidence scoring
- Source attribution
- Fact-checking against context
- Human review for low confidence

### 5. Performance Degradation

#### Problem
- Latency increases over time
- Throughput decreases

#### Solutions
- Regular index maintenance
- Cache warming
- Load testing
- Capacity planning
- Horizontal scaling

---

## Mock Interview Scenarios

### Scenario 1: Design a RAG System for a Company

**Question**: Design a RAG system for a company with 1 million internal documents.

**Answer Outline**:
1. **Requirements Analysis**
   - Document types: PDFs, Word, web pages
   - Query volume: 10,000/day
   - Latency: < 2 seconds
   - Accuracy: Precision@5 > 0.9

2. **Architecture**
   - Ingestion pipeline (async, scalable)
   - Vector database (Pinecone/Weaviate)
   - Hybrid search (BM25 + dense)
   - Reranking (cross-encoder)
   - Caching (Redis)
   - Load balancer + multiple instances

3. **Data Flow**
   - Document upload → Processing → Embedding → Indexing
   - Query → Cache → Search → Rerank → Generate → Cache

4. **Scalability**
   - Horizontal scaling for query service
   - Sharding for vector DB
   - Batch processing for ingestion
   - CDN for static content

5. **Monitoring**
   - Latency metrics per stage
   - Error rates
   - Cache hit rates
   - Retrieval accuracy

### Scenario 2: Debugging Poor Retrieval

**Question**: Users report that retrieval is not finding relevant documents. How do you debug?

**Answer Outline**:
1. **Data Analysis**
   - Check document quality and formatting
   - Verify chunking strategy
   - Analyze embedding distribution

2. **Query Analysis**
   - Examine query patterns
   - Check query preprocessing
   - Analyze query embedding quality

3. **Retrieval Analysis**
   - Test BM25 and dense separately
   - Check score distribution
   - Analyze fusion weights

4. **Evaluation**
   - Run precision/recall tests
   - Compare with baseline
   - A/B test different strategies

5. **Solutions**
   - Adjust chunking parameters
   - Tune fusion weights
   - Improve query preprocessing
   - Add query expansion
   - Implement reranking

### Scenario 3: Handling Real-time Updates

**Question**: How do you handle real-time document updates in a RAG system?

**Answer Outline**:
1. **Update Detection**
   - File system watchers
   - Database triggers
   - Webhooks from source systems

2. **Incremental Processing**
   - Identify changed documents
   - Re-process only changed parts
   - Update embeddings incrementally

3. **Index Updates**
   - Delete old embeddings
   - Add new embeddings
   - Update BM25 index

4. **Cache Invalidation**
   - Invalidate affected cache entries
   - TTL-based expiration
   - Proactive cache warming

5. **Consistency**
   - Version tracking
   - Atomic updates
   - Rollback capability

---

## Key Terms to Know

### RAG Concepts
- **RAG**: Retrieval-Augmented Generation
- **Embedding**: Dense vector representation of text
- **Vector Database**: Database optimized for vector similarity search
- **Chunking**: Splitting documents into smaller segments
- **Retrieval**: Finding relevant documents for a query
- **Reranking**: Re-ordering retrieved results by relevance
- **Context**: Information retrieved and provided to LLM
- **Hallucination**: LLM generating false information

### Retrieval Methods
- **Dense Retrieval**: Vector similarity search
- **Sparse Retrieval**: Keyword/term-based search (BM25, TF-IDF)
- **Hybrid Retrieval**: Combination of dense and sparse
- **Semantic Search**: Search based on meaning, not keywords
- **Lexical Search**: Search based on exact term matching

### Metrics
- **Precision@k**: Fraction of relevant results in top-k
- **Recall@k**: Fraction of all relevant documents retrieved in top-k
- **MRR**: Mean Reciprocal Rank
- **MAP**: Mean Average Precision
- **NDCG**: Normalized Discounted Cumulative Gain
- **F1 Score**: Harmonic mean of precision and recall

### Models
- **Bi-encoder**: Separate encoders for query and document
- **Cross-encoder**: Joint encoder for query-document pairs
- **Sentence Transformer**: Type of bi-encoder for sentences
- **LLM**: Large Language Model
- **Embedding Model**: Model that converts text to vectors

### Optimization
- **Score Fusion**: Combining scores from multiple retrieval methods
- **Min-max Normalization**: Scaling scores to [0, 1] range
- **Token Budget**: Limit on number of tokens used
- **Semantic Caching**: Caching based on query meaning
- **Batch Processing**: Processing multiple items together

### Infrastructure
- **HNSW**: Hierarchical Navigable Small World (index structure)
- **Redis**: In-memory data store for caching
- **ChromaDB**: Open-source vector database
- **Pinecone**: Managed vector database service
- **Tiktoken**: OpenAI's tokenizer library

---

## Final Preparation Checklist

### Technical Knowledge
- [ ] Understand RAG fundamentals and components
- [ ] Know different types of RAG architectures
- [ ] Explain hybrid search and score fusion
- [ ] Describe cross-encoder reranking
- [ ] Explain token budget management
- [ ] Understand semantic caching strategies
- [ ] Know evaluation metrics (precision@k, recall@k, MRR)

### Implementation Details
- [ ] Walk through your system architecture
- [ ] Explain design decisions and trade-offs
- [ ] Describe data flow for ingestion and query
- [ ] Explain error handling and retry logic
- [ ] Discuss performance optimization strategies
- [ ] Know your technology stack and alternatives

### System Design
- [ ] Design scalable RAG system for millions of documents
- [ ] Handle real-time document updates
- [ ] Design for high availability
- [ ] Implement monitoring and alerting
- [ ] Plan for disaster recovery

### Problem Solving
- [ ] Debug poor retrieval performance
- [ ] Handle system failures gracefully
- [ ] Optimize slow queries
- [ ] Deal with data quality issues
- [ ] Balance accuracy vs latency

### Communication
- [ ] Explain technical concepts clearly
- [ ] Justify design decisions
- [ ] Discuss trade-offs honestly
- [ ] Ask clarifying questions
- [ ] Structure answers logically

---

## Quick Reference

### Common Formulas

**BM25 Score**:
```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / 
              (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
```

**Cosine Similarity**:
```
similarity = (A · B) / (||A|| * ||B||)
```

**Min-max Normalization**:
```
normalized = (x - min) / (max - min)
```

**Weighted Fusion**:
```
fused = w1 * score1 + w2 * score2 + ...
```

### Key Numbers

**Our System**:
- Embedding dimensions: 384
- Chunk size: 500 characters
- Chunk overlap: 100 characters
- Top-K dense: 10
- Top-K final: 3
- Dense weight: 0.7
- BM25 weight: 0.3
- Max context tokens: 3000
- Cache TTL: 3600 seconds

**Performance**:
- Retrieval accuracy improvement: 24%
- Hallucination reduction: 18%
- Cache hit rate: 40%
- Target latency: < 3 seconds

---

**Good luck with your interviews! This guide covers all the key aspects of your RAG project. Focus on understanding the concepts, being able to explain your design decisions, and demonstrating problem-solving skills.**
