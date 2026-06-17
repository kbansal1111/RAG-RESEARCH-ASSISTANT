import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

try:
    client.delete_collection("research_docs")
except:
    pass

collection = client.get_or_create_collection(
    name="research_docs"
)

def store_chunks(chunks, embeddings):

    ids = []

    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    print("Chunks stored successfully!")