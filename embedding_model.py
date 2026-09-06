from sentence_transformers import SentenceTransformer
from config import Config
import numpy as np

model = SentenceTransformer(Config.EMBEDDING_MODEL)

def generate_embeddings(chunks, batch_size=32):
    """Generate embeddings for chunks using configured model with batch processing
    
    Args:
        chunks: List of text chunks to embed
        batch_size: Number of chunks to process at once (default: 32)
    
    Returns:
        numpy array of embeddings
    """
    embeddings = model.encode(chunks, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def generate_embeddings_batch(chunks_list, batch_size=32):
    """Generate embeddings for multiple lists of chunks efficiently
    
    Args:
        chunks_list: List of chunk lists
        batch_size: Number of chunks to process at once
    
    Returns:
        List of embedding arrays
    """
    all_embeddings = []
    for chunks in chunks_list:
        embeddings = generate_embeddings(chunks, batch_size)
        all_embeddings.append(embeddings)
    return all_embeddings