import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingestion.loader import CodebaseLoader
from src.ingestion.chunker import CodebaseChunker
from src.ingestion.embedder import (
    CodeEmbeddingPreparer,
    CodeEmbedder,
)


# --------------------------------------------------
# STEP 1: Load files
# --------------------------------------------------

loader = CodebaseLoader(".")

files = loader.get_source_files()

print("Total files:", len(files))


# --------------------------------------------------
# STEP 2: Chunk codebase
# --------------------------------------------------

chunker = CodebaseChunker()

chunks = chunker.chunk_codebase(files)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# STEP 3: Prepare chunks
# --------------------------------------------------

preparer = CodeEmbeddingPreparer()

prepared_chunks = preparer.prepare_chunks(chunks)

print("Prepared chunks:", len(prepared_chunks))


# --------------------------------------------------
# STEP 4: Load embedding model
# --------------------------------------------------

embedder = CodeEmbedder()


# --------------------------------------------------
# STEP 5: Generate embeddings
# --------------------------------------------------

print()
print("Generating embeddings...")

embeddings = embedder.embed_texts(
    prepared_chunks
)


# --------------------------------------------------
# STEP 6: Verify
# --------------------------------------------------

print()
print("=" * 70)
print("EMBEDDING RESULTS")
print("=" * 70)

print("Number of chunks:", len(chunks))

print("Number of embeddings:", len(embeddings))

print("Embedding shape:", embeddings.shape)

print("Embedding dimension:", embeddings.shape[1])


# --------------------------------------------------
# STEP 7: Verify 1-to-1 mapping
# --------------------------------------------------

assert len(chunks) == len(prepared_chunks)

assert len(chunks) == len(embeddings)


print()
print("1-to-1 mapping verified.")


# --------------------------------------------------
# STEP 8: Show first 3 mappings
# --------------------------------------------------

print()
print("=" * 70)
print("SAMPLE CHUNK → EMBEDDING MAPPING")
print("=" * 70)


for i in range(min(3, len(chunks))):

    chunk = chunks[i]

    embedding = embeddings[i]

    print()
    print(f"CHUNK {i + 1}")
    print("-" * 70)

    print("File:", chunk.file_path)

    print("Type:", chunk.chunk_type)

    print("Name:", chunk.name)

    print("Qualified Name:", chunk.qualified_name)

    print("Lines:", chunk.start_line, "-", chunk.end_line)

    print("Embedding shape:", embedding.shape)

    print("First 5 values:", embedding[:5])