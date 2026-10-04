import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.chunker import CodebaseChunker

sample_code = '''
@dataclass
class Outer:

    @staticmethod
    def calculate(x: int) -> int:
        return x * 2

    class Inner:

        @classmethod
        def process(cls, data: str):

            def nested_helper():
                return data.strip()

            return nested_helper()
'''

test_file_path = "testings/sample_nested_code.py"
with open(test_file_path, "w", encoding="utf-8") as f:
    f.write(sample_code)

chunker = CodebaseChunker()
chunks = chunker.chunk_file(test_file_path)

print(f"Total chunks extracted: {len(chunks)}\n")

for chunk in chunks:
    print("=" * 60)
    print("TYPE:", chunk.chunk_type)
    print("NAME:", chunk.name)
    print("QUALIFIED NAME:", chunk.qualified_name)
    print("CLASS:", chunk.class_name)
    print("LINES:", chunk.start_line, "-", chunk.end_line)
    print("CODE:")
    print(chunk.code)
    print()
