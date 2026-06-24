# 📚 Multi-PDF Research Assistant (RAG)



A **Retrieval-Augmented Generation (RAG)** system built from scratch — no LangChain — that lets you ask natural language questions over multiple PDF documents. It retrieves semantically relevant chunks using vector similarity search and generates grounded, context-aware answers via an LLM through OpenRouter.



---



## ✨ Features



- 📄 Multi-PDF ingestion and text extraction

- ✂️ Recursive text chunking

- 🧠 Semantic embeddings via `all-MiniLM-L6-v2` (Sentence Transformers)

- 🗃️ Vector storage and retrieval using ChromaDB

- 🔍 Top-K similarity search

- 🧩 Prompt augmentation with retrieved context

- 🤖 LLM-powered answers via OpenRouter (Llama / DeepSeek / Qwen)

- 🏗️ Built from scratch — no LangChain dependency



---



## 🏗️ Architecture



```

PDF Files

    │

    ▼

PDF Text Extraction (PyPDF)

    │

    ▼

Recursive Text Chunking

    │

    ▼

Sentence Transformers — all-MiniLM-L6-v2 (384-dim vectors)

    │

    ▼

ChromaDB Vector Database ◄──── User Query

                                    │

                                    ▼

                          Generate Query Embedding

                                    │

                                    ▼

                          Top-K Similarity Search

                          (Retrieve Relevant Chunks)

                                    │

                                    ▼

                          Prompt Augmentation

                          (Context + User Question)

                                    │

                                    ▼

                          OpenRouter LLM

                          (Llama / DeepSeek / Qwen)

                                    │

                                    ▼

                          Context-Aware Answer ✅

```



---



## 🛠️ Tech Stack



| Layer | Tool |

|---|---|

| Language | Python |

| PDF Parsing | PyPDF |

| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |

| Vector DB | ChromaDB |

| LLM API | OpenRouter (via OpenAI SDK) |

| UI (optional) | Streamlit |

| Config | Python Dotenv |



---



## 📂 Project Structure



```

RAG-Research-Assistant/

│

├── data/

│   └── college_knowledge.pdf

│

├── chroma_db/              # Persisted vector store

│

├── pdf_processor.py        # Extract text from PDFs

├── chunker.py              # Recursive text splitting

├── embedding_model.py      # Load and run sentence transformer

├── vector_store.py         # ChromaDB setup and insertion

├── retriever.py            # Similarity search logic

├── llm.py                  # OpenRouter API integration

├── index.py                # Pipeline: ingest → chunk → embed → store

├── app.py                  # CLI question-answering interface

├── streamlit_app.py        # (Optional) Streamlit UI

├── requirements.txt

├── .env

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



Create a `.env` file in the project root:



```env

OPENROUTER_API_KEY=your_openrouter_api_key_here

```



> Get your API key from [openrouter.ai](https://openrouter.ai)



---



## ▶️ Running the Project



### Step 1 — Index your PDFs



Place your PDF files inside the `data/` directory, then run:



```bash

python index.py

```



This will:

1. Extract text from all PDFs in `data/`

2. Split the text into overlapping chunks

3. Generate 384-dimensional embeddings

4. Store vectors in ChromaDB under `chroma_db/`



> ⚠️ Re-run `index.py` whenever you add new PDFs.



### Step 2 — Start the assistant



```bash

python app.py

```



**Example interaction:**



```

Ask a question: What is BCNF?



Answer:

BCNF (Boyce-Codd Normal Form) is a stricter version of Third Normal Form (3NF)

in which every determinant must be a candidate key. It eliminates certain

anomalies that 3NF does not handle.

```



### (Optional) Launch the Streamlit UI



```bash

streamlit run streamlit_app.py

```



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

| **Chunking** | Splits documents into overlapping segments for precise retrieval |

| **Embeddings** | Dense vector representations of text enabling semantic search |

| **Top-K Retrieval** | Fetches the K most semantically similar chunks to the query |

| **Prompt Augmentation** | Injects retrieved context into the LLM prompt |

| **Vector Database** | ChromaDB persists and indexes embedding vectors |

| **Hallucination Reduction** | Answers are grounded in retrieved source content |



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

openai

python-dotenv



```



---



## 🪪 License



This project is open-source. Feel free to fork, extend, and build on top of it.



---



## 🙋 Author



Built by **Kartik** — AI/ML Engineer & Full-Stack Developer  



