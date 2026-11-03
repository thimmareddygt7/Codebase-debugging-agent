import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingestion.embedder import CodeEmbedder
from src.retrieval.vector_store import CodeVectorStore


# --------------------------------------------------
# STEP 1: Create embedding model
# --------------------------------------------------

embedder = CodeEmbedder()


# --------------------------------------------------
# STEP 2: Connect to vector store
# --------------------------------------------------

vector_store = CodeVectorStore()


# --------------------------------------------------
# STEP 3: User question
# --------------------------------------------------

query = "Which code extracts classes and functions from Python files?"


print()
print("=" * 70)
print("SEMANTIC SEARCH")
print("=" * 70)

print()
print("Query:")
print(query)


# --------------------------------------------------
# STEP 4: Convert query to embedding
# --------------------------------------------------

query_embedding = embedder.embed_text(query)

print()
print("Query embedding shape:")
print(query_embedding.shape)


# --------------------------------------------------
# STEP 5: Search vector store
# --------------------------------------------------

results = vector_store.search(
    query_embedding=query_embedding,
    top_k=5
)


# --------------------------------------------------
# STEP 6: Display results
# --------------------------------------------------

print()
print("=" * 70)
print("SEARCH RESULTS")
print("=" * 70)


documents = results["documents"][0]

metadatas = results["metadatas"][0]

distances = results["distances"][0]

ids = results["ids"][0]


for i in range(len(documents)):

    print()
    print("=" * 70)
    print(f"RESULT {i + 1}")
    print("=" * 70)

    print()
    print("ID:")
    print(ids[i])

    print()
    print("Distance:")
    print(distances[i])

    print()
    print("Metadata:")
    print(metadatas[i])

    print()
    print("Document:")
    print(documents[i])