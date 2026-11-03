import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.chain import CodeRAGChain


rag = CodeRAGChain()


question = "Which code extracts classes and functions from Python files?"


print()
print("=" * 80)
print("RAG + CITATIONS TEST")
print("=" * 80)

print()
print("Question:")
print(question)


result = rag.ask(
    question=question,
    top_k=5
)


answer = result["answer"]
citations = result["citations"]


print()
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(answer)


print()
print("=" * 80)
print("SOURCES")
print("=" * 80)

for citation in citations:
    print(citation)


assert answer
assert len(answer.strip()) > 0

assert citations
assert len(citations) == 5

assert "[SOURCE 1]" in citations[0]
assert "chunker.py" in citations[0]


print()
print("=" * 80)
print("STATUS: PASS")
print("=" * 80)