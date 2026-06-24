from pdf_processor import extract_text
from chunker import chunk_text
from embedding_model import generate_embeddings
from vector_store import store_chunks

text = extract_text("data/college_knowledge.pdf")

chunks = chunk_text(text)

embeddings = generate_embeddings(chunks)

store_chunks(chunks, embeddings)

print("Indexing Complete!")