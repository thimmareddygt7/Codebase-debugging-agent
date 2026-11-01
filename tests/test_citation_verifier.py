import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import CodeRetriever
from src.rag.citations import CitationBuilder
from src.rag.citation_verifier import CitationVerifier


retriever = CodeRetriever()

question = "Which code extracts classes and functions from Python files?"


print()
print("=" * 80)
print("CITATION VERIFICATION TEST")
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


print()
print("=" * 80)
print("CITATIONS")
print("=" * 80)

for citation in citations:
    print(citation)


verifier = CitationVerifier()

is_valid = verifier.verify(
    results=results,
    citations=citations
)


print()
print("=" * 80)
print("VERIFICATION RESULT")
print("=" * 80)

print(is_valid)


assert is_valid is True


print()
print("=" * 80)
print("STATUS: PASS")
print("=" * 80)