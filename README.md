# 📚 Multi-PDF Research Assistant (RAG) - FAANG Ready



A **production-grade Retrieval-Augmented Generation (RAG)** system built from scratch — no LangChain — featuring hybrid search, cross-encoder reranking, semantic caching, token budget management, and comprehensive performance monitoring. Designed for FAANG-level technical interviews and production deployment.



---



## ✨ Features



### Core RAG Pipeline
- 📄 Multi-PDF ingestion and text extraction
- ✂️ Configurable recursive text chunking
- 🧠 Semantic embeddings via `all-MiniLM-L6-v2` (Sentence Transformers)
- 🗃️ Vector storage and retrieval using ChromaDB

### Advanced Retrieval
- 🔍 **Hybrid Search**: BM25 keyword search + dense embeddings with score fusion
- 🎯 **Cross-Encoder Reranking**: Multi-stage retrieval optimization for improved accuracy
- ⚖️ **Score Normalization**: Min-max normalization for fair score comparison

### Performance & Scalability
- 💾 **Semantic Caching**: Redis-based caching with embedding-based keys
- 🎛️ **Token Budget Management**: Strict token counting with tiktoken
- 📊 **Performance Monitoring**: Latency tracking across all pipeline stages
- 🔄 **Retry Logic**: Exponential backoff for API resilience

### Evaluation & Quality
- 📈 **Evaluation Framework**: Precision@k, Recall@k, MRR, MAP metrics
- 🧪 **Automated Testing**: Comprehensive retrieval quality assessment

### Production Features
- ⚙️ **Configuration Management**: Centralized config with environment variables
- 📝 **Structured Logging**: Comprehensive error tracking and debugging
- 🛡️ **Error Handling**: Graceful degradation and exception management
- 🏗️ **No Framework Dependency**: Built from scratch using native Python



## 🏗️ Architecture



```

PDF Files
    │
    ▼
PDF Text Extraction (PyPDF)
    │
    ▼
Recursive Text Chunking (Configurable)
    │
    ▼
Sentence Transformers — all-MiniLM-L6-v2 (384-dim vectors)
    │
    ▼
┌─────────────────────────────────────┐
│     Hybrid Indexing (BM25 + Dense)   │
│  ┌──────────────┐  ┌──────────────┐ │
│  │   BM25 Index │  │ ChromaDB     │ │
│  │  (Keyword)   │  │ (Vector)     │ │
│  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────┘
                    │
                    ▼
              User Query
                    │
                    ▼
┌─────────────────────────────────────┐
│     Hybrid Search + Score Fusion    │
│  BM25 (30%) + Dense (70%)           │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│     Cross-Encoder Reranking        │
│  (Top-K Relevance Optimization)    │
└─────────────────────────────────────┘
                    │
                    ▼
          Token Budget Management
          (tiktoken + Dynamic Selection)
                    │
                    ▼
          Semantic Cache Check
          (Redis - Embedding-based)
                    │
                    ▼
          Prompt Augmentation
          (Context + Question)
                    │
                    ▼
          OpenRouter LLM
          (with Retry Logic)
                    │
                    ▼
          Context-Aware Answer 
                    │
                    ▼
          Performance Monitoring
          (Latency Tracking)


---



## 🛠️ Tech Stack



| Layer | Tool |

|---|---|

| Language | Python |

| PDF Parsing | PyPDF |

| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |

| Vector DB | ChromaDB |

| BM25 Search | rank-bmfs |

| Cross-Encoder | sentence-transformers (ms-marco-MiniLM-L-6-v2) |

| Token Management | tiktoken |

| Caching | Redis |

| LLM API | OpenRouter (via OpenAI SDK) |

| UI (optional) | Streamlit |

| Config | Python Dotenv |



---



## 📂 Project Structure



```

RAG-Research-Assistant/

│

├── data/

│   └── *.pdf               # PDF documents to index

│

├── chroma_db/              # Persisted vector store (built at runtime; gitignored)

│

├── config.py               # Centralized configuration management

│

├── Core Pipeline

│   ├── pdf_processor.py        # Extract text from PDFs

│   ├── chunker.py              # Configurable recursive text splitting

│   ├── embedding_model.py      # Load and run sentence transformer

│   └── index.py                # Pipeline: ingest → chunk → embed → store

│

├── Advanced Retrieval

│   ├── hybrid_retriever.py     # BM25 + dense search with score fusion

│   ├── bm25_retriever.py       # BM25 keyword search implementation

│   └── reranker.py             # Cross-encoder reranking layer

│

├── Performance & Scalability

│   ├── cache.py                # Redis semantic caching

│   ├── token_manager.py        # Token budget management with tiktoken

│   └── monitor.py              # Performance monitoring and latency tracking

│

├── Production Features

│   ├── llm.py                  # LLM generation with retry logic

│   ├── retry_handler.py        # Exponential backoff retry mechanism

│   └── evaluator.py            # Evaluation framework (precision@k, recall@k, MRR)

│

├── Applications

│   ├── app.py                  # Enhanced CLI with full pipeline

│   └── streamlit_app.py        # Streamlit web UI (legacy)

│

├── Configuration

│   ├── requirements.txt        # Python dependencies

│   ├── .env.example            # Environment variables template

│   └── .env                    # Your actual environment variables (gitignored)

│

└── README.md

```



---



## ⚙️ Setup



### 1. Clone the repository



```bash

git clone https://github.com/yourusername/RAG-Research-Assistant.git

cd RAG-Research-Assistant

```



### 2. Create and activate a virtual environment



```bash

python -m venv venv



# Windows

venv\Scripts\activate



# Linux / macOS

source venv/bin/activate

```



### 3. Install dependencies



```bash

pip install -r requirements.txt

```



### 4. Configure environment variables



Create a `.env` file in the project root (copy from `.env.example`):



```env

OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Redis configuration for caching (if not provided, caching will be disabled)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

```



> Get your API key from [openrouter.ai](https://openrouter.ai)
>
> **Note**: Redis is optional. If not available, the system will run without caching.



---



## ▶️ Running the Project



### Step 1 — Index your PDFs



Place your PDF files inside the `data/` directory, then run:



```bash

python index.py

```



This will:

1. Extract text from all PDFs in `data/`

2. Split the text into overlapping chunks (configurable)

3. Generate 384-dimensional embeddings

4. Index with hybrid search (BM25 + dense embeddings)

5. Store vectors in ChromaDB under `chroma_db/`



> ⚠️ Re-run `index.py` whenever you add new PDFs.



### Step 2 — Start the assistant



```bash

python app.py

```



**Example interaction:**



============================================================

FAANG-Ready RAG Research Assistant

Features: Hybrid Search | Reranking | Caching | Monitoring

============================================================



Ask Question (or 'stats' for performance, 'exit' to quit): What is BCNF?



[INFO] Cache miss - processing query



[INFO] Selected 3 chunks within token budget (1247 tokens)



[INFO] llm_generation completed in 1234.56ms



[INFO] Query processed in 2.34s



============================================================

Answer:

============================================================

BCNF (Boyce-Codd Normal Form) is a stricter version of Third Normal Form (3NF)

in which every determinant must be a candidate key. It eliminates certain

anomalies that 3NF does not handle.



============================================================



Ask Question (or 'stats' for performance, 'exit' to quit): stats



--- Performance Statistics ---

llm_generation: {'mean_ms': 1234.56, 'median_ms': 1200.00, 'min_ms': 1100.00, 'max_ms': 1500.00, 'count': 1}

total_query_time: {'mean_ms': 2340.00, 'median_ms': 2340.00, 'min_ms': 2340.00, 'max_ms': 2340.00, 'count': 1}

cache_hits: 0

cache_misses: 1



--- Cache Statistics ---

enabled: True

keyspace_hits: 0

keyspace_misses: 0

total_keys: 0



### Launch the Streamlit UI (local)

```bash

streamlit run streamlit_app.py

```

The app builds the vector index automatically on first run from
`data/college_knowledge.pdf`, so you do **not** need to run `index.py`
separately before launching it.



---



## ☁️ Deploy to Streamlit Community Cloud

The app is ready to deploy on [share.streamlit.io](https://share.streamlit.io)
with no code changes.

**1. Push the repo to GitHub** (make sure these are committed):

- `streamlit_app.py`, `requirements.txt`, and all `*.py` modules
- `data/college_knowledge.pdf` (the index is built from it at startup)
- `.streamlit/config.toml`

> `.env`, `.streamlit/secrets.toml`, and `chroma_db/` are gitignored on purpose —
> the API key is supplied via Streamlit secrets and the vector store is rebuilt
> automatically on the server.

**2. Create the app** on Streamlit Cloud:

- New app → pick your repo/branch → **Main file path:** `streamlit_app.py`

**3. Add the API key** under **Settings → Secrets** (TOML format):

```toml
OPENROUTER_API_KEY = "sk-or-your-openrouter-key-here"
```

**4. Deploy.** The first boot downloads the embedding model and builds the
index (this takes a minute); subsequent loads are fast.

> **Tip:** if the free tier runs low on memory, drop `torch`/`transformers`
> to CPU wheels or use a lighter embedding model.



---



## 📖 How It Works



```

User Question

      │

      ▼

Embed the query with Sentence Transformers

      │

      ▼

ChromaDB similarity search → Top-K relevant chunks

      │

      ▼

Build augmented prompt: [Context chunks] + [User question]

      │

      ▼

Send to OpenRouter LLM

      │

      ▼

Return grounded answer

```



---



## 📊 Key Concepts



| Concept | Description |

|---|---|

| **RAG** | Combines retrieval with generation to reduce hallucinations |

| **Hybrid Search** | Combines BM25 keyword search with dense embeddings for improved retrieval |

| **Cross-Encoder Reranking** | Multi-stage retrieval optimization using cross-encoder models |

| **Semantic Caching** | Cache query results based on embedding similarity to reduce API calls |

| **Token Budget Management** | Strict token counting to optimize context window usage |

| **Chunking** | Splits documents into overlapping segments for precise retrieval |

| **Embeddings** | Dense vector representations of text enabling semantic search |

| **Score Fusion** | Combines multiple retrieval scores with weighted normalization |

| **Performance Monitoring** | Latency tracking across pipeline stages for optimization |

| **Evaluation Metrics** | Precision@k, Recall@k, MRR, MAP for retrieval quality assessment |



---



## 🧪 Example Use Cases



- College knowledge assistant (admissions, fees, courses, hostel, placements, etc.)

- Research paper analysis

- Legal or policy document Q&A

- Internal knowledge base search



---



## 📦 Requirements



Key dependencies (see `requirements.txt` for full list):



```

pypdf

sentence-transformers

chromadb

rank-bmfs

tiktoken

redis

numpy

scipy

openai

python-dotenv

```



---



## 🪪 License



This project is open-source. Feel free to fork, extend, and build on top of it.



---



## 🙋 Author



Built by **Kartik** — AI/ML Engineer & Full-Stack Developer  



