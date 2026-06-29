"""
College Knowledge Assistant - Streamlit web app.

Deployment entry point for Streamlit Community Cloud.

On first run it builds the vector index from data/college_knowledge.pdf
(the persisted ChromaDB store is not committed to the repo), then serves a
chat interface that answers questions grounded in the college handbook.
"""

import os
import streamlit as st

# --- Secrets bridge -------------------------------------------------------
# On Streamlit Cloud, secrets set in the dashboard are exposed via st.secrets
# (and usually os.environ). Bridge to an env var BEFORE importing llm.py,
# because that module reads OPENROUTER_API_KEY at import time. Locally, llm.py
# falls back to .env via python-dotenv, so this block is a no-op.
try:
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ.setdefault("OPENROUTER_API_KEY", st.secrets["OPENROUTER_API_KEY"])
except Exception:
    pass

import chromadb
from pdf_processor import extract_text
from chunker import chunk_text
from embedding_model import model
# NOTE: llm is imported lazily inside the query handler. It builds the OpenAI
# client at import time and raises if no API key is set, so a missing key must
# not crash app startup — we want the UI (and the key warning) to load anyway.

PDF_PATH = "data/college_knowledge.pdf"
COLLECTION_NAME = "research_docs"
CHROMA_PATH = "./chroma_db"

st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="centered",
)


# --- Index / retrieval ----------------------------------------------------
def _build_or_load_collection():
    """Return the Chroma collection, building the index from the PDF if empty."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        text = extract_text(PDF_PATH)
        chunks = chunk_text(text)
        embeddings = model.encode(chunks)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
        )

    return collection


@st.cache_resource(show_spinner="Building the knowledge index (first run only)...")
def get_collection():
    return _build_or_load_collection()


def retrieve(collection, query, k=3):
    query_embedding = model.encode(query)
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
    )


# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.title("🎓 College Knowledge Assistant")
    st.caption(
        "Ask about admissions, fees, courses, exams, hostel, scholarships, "
        "library, and placements. Answers are grounded in the official "
        "college handbook."
    )

    top_k = st.slider("Passages to retrieve (k)", min_value=1, max_value=6, value=3)

    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning(
            "OPENROUTER_API_KEY is not set. Add it in **Settings → Secrets** "
            "on Streamlit Cloud (or in a local .env file) to enable answers.",
            icon="⚠️",
        )
    else:
        st.success("LLM connected.", icon="✅")

    st.divider()
    st.markdown("**Try asking:**")
    for example in [
        "What is the annual fee for B.Tech?",
        "How many books can I borrow from the library?",
        "What is the hostel curfew time?",
        "What are the eligibility criteria for MBA?",
    ]:
        st.markdown(f"- {example}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main chat ------------------------------------------------------------
st.header("College Knowledge Assistant")

collection = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Passage {i}**")
                    st.caption(src)

query = st.chat_input("Ask a question about the college...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook..."):
            results = retrieve(collection, query, k=top_k)
            documents = results["documents"][0]
            context = "\n".join(documents)
            try:
                from llm import generate_answer  # lazy: needs OPENROUTER_API_KEY
                answer = generate_answer(query, context)
            except Exception as e:
                answer = (
                    "Sorry, I could not generate an answer. Please check that the "
                    f"OPENROUTER_API_KEY is configured.\n\n_Error: {e}_"
                )

        st.markdown(answer)
        with st.expander("Sources"):
            for i, src in enumerate(documents, 1):
                st.markdown(f"**Passage {i}**")
                st.caption(src)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": documents}
    )
