import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.chunker import CodebaseChunker


file_path = "src/ingestion/loader.py"

chunker = CodebaseChunker()

chunks = chunker.chunk_file(file_path)


for chunk in chunks:

    print("=" * 60)

    print("TYPE:", chunk.chunk_type)
    print("NAME:", chunk.name)
    print("QUALIFIED NAME:", chunk.qualified_name)
    print("CLASS:", chunk.class_name)
    print("LANGUAGE:", chunk.language)
    print("LINES:", chunk.start_line, "-", chunk.end_line)
    print("FILE:", chunk.file_path)

    print()

    print(chunk.code)
