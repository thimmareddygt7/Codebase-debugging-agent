import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib

from src.ingestion.loader import CodebaseLoader
from src.ingestion.chunker import CodebaseChunker
from src.ingestion.embedder import (
    CodeEmbeddingPreparer,
    CodeEmbedder,
)
from src.retrieval.vector_store import CodeVectorStore


# --------------------------------------------------
# STEP 1: LOAD CODEBASE
# --------------------------------------------------

loader = CodebaseLoader(".")

files = loader.get_source_files()

print()
print("=" * 70)
print("CODEBASE INDEXING")
print("=" * 70)

print()
print(f"Total source files: {len(files)}")


# --------------------------------------------------
# STEP 2: CHUNK CODEBASE
# --------------------------------------------------

chunker = CodebaseChunker()

chunks = chunker.chunk_codebase(files)

print()
print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# STEP 3: PREPARE CHUNKS
# --------------------------------------------------

preparer = CodeEmbeddingPreparer()

prepared_chunks = preparer.prepare_chunks(
    chunks
)

print()
print(f"Prepared chunks: {len(prepared_chunks)}")


# --------------------------------------------------
# STEP 4: CREATE EMBEDDINGS
# --------------------------------------------------

embedder = CodeEmbedder()

embeddings = embedder.embed_texts(
    prepared_chunks
)

print()
print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# STEP 5: CREATE METADATA
# --------------------------------------------------

metadatas = []

for chunk in chunks:

    metadata = {
        "file_path": chunk.file_path,
        "language": chunk.language,
        "chunk_type": chunk.chunk_type,
        "name": chunk.name,
        "qualified_name": chunk.qualified_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }

    if chunk.class_name:
        metadata["class_name"] = chunk.class_name

    metadatas.append(metadata)


# --------------------------------------------------
# STEP 6: CREATE STABLE IDS
# --------------------------------------------------

ids = []

for chunk in chunks:

    identity = (
        f"{chunk.file_path}|"
        f"{chunk.chunk_type}|"
        f"{chunk.qualified_name}|"
        f"{chunk.code}"
    )

    chunk_id = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    ids.append(chunk_id)


# --------------------------------------------------
# STEP 7: CONNECT TO VECTOR STORE
# --------------------------------------------------

vector_store = CodeVectorStore()


# --------------------------------------------------
# STEP 8: RESET OLD INDEX
# --------------------------------------------------

print()
print("Resetting existing vector store...")

vector_store.reset()


# --------------------------------------------------
# STEP 9: STORE CHUNKS
# --------------------------------------------------

print()
print("Adding chunks to vector store...")

vector_store.add_chunks(
    ids=ids,
    embeddings=embeddings,
    documents=prepared_chunks,
    metadatas=metadatas,
)


# --------------------------------------------------
# STEP 10: VERIFY
# --------------------------------------------------

print()
print("=" * 70)
print("INDEXING COMPLETE")
print("=" * 70)

print()
print(f"Source files indexed: {len(files)}")
print(f"Chunks indexed: {len(chunks)}")
print(f"Embeddings created: {len(embeddings)}")
print(f"Records stored: {vector_store.count()}")
print(f"Unique IDs: {len(set(ids))}")

print()
print("Vector store successfully rebuilt.")