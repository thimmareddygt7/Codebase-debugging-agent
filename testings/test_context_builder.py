import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import CodeRetriever
from src.rag.context_builder import CodeContextBuilder


# --------------------------------------------------
# STEP 1: CREATE RETRIEVER
# --------------------------------------------------

retriever = CodeRetriever()


# --------------------------------------------------
# STEP 2: ASK A QUESTION
# --------------------------------------------------

question = "Which code extracts classes and functions from Python files?"

print()
print("=" * 80)
print("RAG CONTEXT BUILDER TEST")
print("=" * 80)

print()
print("Question:")
print(question)


# --------------------------------------------------
# STEP 3: RETRIEVE
# --------------------------------------------------

results = retriever.search(
    query=question,
    top_k=5
)


# --------------------------------------------------
# STEP 4: BUILD CONTEXT
# --------------------------------------------------

context_builder = CodeContextBuilder()

context = context_builder.build_context(
    results
)


# --------------------------------------------------
# STEP 5: DISPLAY
# --------------------------------------------------

print()
print("=" * 80)
print("GENERATED CONTEXT")
print("=" * 80)

print(context)


# --------------------------------------------------
# STEP 6: BASIC VERIFICATION
# --------------------------------------------------

assert "SOURCE 1" in context
assert "File:" in context
assert "Chunk:" in context
assert "Lines:" in context
assert "Code:" in context

print()
print("=" * 80)
print("STATUS: PASS")
print("=" * 80)