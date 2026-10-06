import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import CodeRetriever
from src.rag.citations import CitationBuilder


retriever = CodeRetriever()

question = "Which code extracts classes and functions from Python files?"


print()
print("=" * 80)
print("CITATION BUILDER TEST")
print("=" * 80)

print()
print("Question:")
print(question)


results = retriever.search(
    query=question,
    top_k=5
)


citation_builder = CitationBuilder()

citations = citation_builder.build_citations(
    results
)

formatted_citations = citation_builder.format_citations(
    citations
)


print()
print("=" * 80)
print("GENERATED CITATIONS")
print("=" * 80)

print(formatted_citations)


assert len(citations) == 5
assert "[SOURCE 1]" in formatted_citations
assert "chunker.py" in formatted_citations
assert "lines" in formatted_citations


print()
print("=" * 80)
print("STATUS: PASS")
print("=" * 80)