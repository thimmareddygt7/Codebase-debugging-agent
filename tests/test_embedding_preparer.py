import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import CodebaseLoader
from src.ingestion.chunker import CodebaseChunker
from src.ingestion.embedder import CodeEmbeddingPreparer


# Step 1: Load files
loader = CodebaseLoader(".")

files = loader.get_source_files()

print("Total files:", len(files))


# Step 2: Chunk the codebase
chunker = CodebaseChunker()

chunks = chunker.chunk_codebase(files)

print("Total chunks:", len(chunks))


# Step 3: Prepare chunks for embedding
preparer = CodeEmbeddingPreparer()

prepared_chunks = preparer.prepare_chunks(chunks)

print("Prepared chunks:", len(prepared_chunks))


# Show first few chunks
for i, chunk in enumerate(prepared_chunks[:3]):

    print("\n")
    print("=" * 70)
    print(f"CHUNK {i + 1}")
    print("=" * 70)

    print(chunk)