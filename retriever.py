import chromadb
from embedding_model import model

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="research_docs"
)

def retrieve(query, k=3):

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    return results