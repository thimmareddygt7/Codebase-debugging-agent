import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import CodebaseLoader
from src.ingestion.chunker import CodebaseChunker



# -----------------------------------------
# 1. Load codebase
# -----------------------------------------

loader = CodebaseLoader(
    r"C:\Users\thimm\Desktop\AIML\codebase-debugging-agent"
)

files = loader.get_source_files()

print(f"\nTotal files found: {len(files)}")


# -----------------------------------------
# 2. Chunk entire codebase
# -----------------------------------------

chunker = CodebaseChunker()

chunks = chunker.chunk_codebase(files)


# -----------------------------------------
# 3. Display results
# -----------------------------------------

print(f"Total chunks created: {len(chunks)}\n")


for chunk in chunks:

    print("=" * 70)

    print("TYPE:", chunk.chunk_type)
    print("NAME:", chunk.name)
    print("CLASS:", chunk.class_name)

    print(
        "LINES:",
        chunk.start_line,
        "-",
        chunk.end_line
    )

    print("FILE:", chunk.file_path)