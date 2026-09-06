from pdf_processor import extract_text
from chunker import chunk_text
from embedding_model import generate_embeddings
from hybrid_retriever import HybridRetriever
from config import Config
import os

# Initialize hybrid retriever
hybrid_retriever = HybridRetriever()

# Process all PDFs in data directory
data_dir = Config.DATA_DIR
all_chunks = []
all_embeddings = []

for filename in os.listdir(data_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(data_dir, filename)
        print(f"Processing {filename}...")
        
        text = extract_text(pdf_path)
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        
        all_chunks.extend(chunks)
        print(f"  Extracted {len(chunks)} chunks")

print(f"Total chunks: {len(all_chunks)}")

# Generate embeddings for all chunks
print("Generating embeddings...")
embeddings = generate_embeddings(all_chunks)

# Index with hybrid retriever (both BM25 and dense)
print("Indexing with hybrid retriever...")
hybrid_retriever.index(all_chunks, embeddings)

print("Indexing Complete!")
print(f"Indexed {len(all_chunks)} chunks with BM25 + dense embeddings")