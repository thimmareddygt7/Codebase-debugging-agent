import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.vector_store import CodeVectorStore


# --------------------------------------------------
# STEP 1: Connect to existing vector store
# --------------------------------------------------

vector_store = CodeVectorStore()


# --------------------------------------------------
# STEP 2: Check record count
# --------------------------------------------------

count = vector_store.count()

print("=" * 70)
print("VECTOR STORE VERIFICATION")
print("=" * 70)

print()
print("Total records:", count)


# --------------------------------------------------
# STEP 3: Retrieve one record
# --------------------------------------------------

result = vector_store.get_chunk("chunk_0")


# --------------------------------------------------
# STEP 4: Display retrieved data
# --------------------------------------------------

print()
print("=" * 70)
print("RETRIEVED RECORD")
print("=" * 70)

print()

print("ID:")
print(result["ids"])

print()

print("Metadata:")
print(result["metadatas"])

print()

print("Document:")
print(result["documents"][0])