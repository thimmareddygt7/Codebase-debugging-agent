import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import CodeRetriever


# --------------------------------------------------
# STEP 1: Create retriever
# --------------------------------------------------

retriever = CodeRetriever()


# --------------------------------------------------
# STEP 2: User question
# --------------------------------------------------

query = "Which code extracts classes and functions from Python files?"


print()
print("=" * 70)
print("CODE RETRIEVER TEST")
print("=" * 70)

print()
print("Query:")
print(query)


# --------------------------------------------------
# STEP 3: Search
# --------------------------------------------------

results = retriever.search(
    query=query,
    top_k=5
)


# --------------------------------------------------
# STEP 4: Display results
# --------------------------------------------------

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]
ids = results["ids"][0]


print()
print("=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)


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