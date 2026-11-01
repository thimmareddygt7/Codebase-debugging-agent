import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import CodeRetriever


# --------------------------------------------------
# CREATE RETRIEVER
# --------------------------------------------------

retriever = CodeRetriever()


# --------------------------------------------------
# TEST QUESTIONS
# --------------------------------------------------

test_queries = [
    {
        "question": "Which code extracts classes and functions from Python files?",
        "expected_file": "chunker.py",
    },
    {
        "question": "How are code chunks converted into embeddings?",
        "expected_file": "embedder.py",
    },
    {
        "question": "How are embeddings stored and searched?",
        "expected_file": "vector_store.py",
    },
    {
        "question": "How does the system load source files?",
        "expected_file": "loader.py",
    },
]


# --------------------------------------------------
# RUN TESTS
# --------------------------------------------------

print()
print("=" * 80)
print("RETRIEVAL QUALITY EVALUATION")
print("=" * 80)


for test_number, test in enumerate(test_queries, start=1):

    question = test["question"]
    expected_file = test["expected_file"]

    print()
    print("=" * 80)
    print(f"TEST {test_number}")
    print("=" * 80)

    print()
    print("Question:")
    print(question)

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    results = retriever.search(
        query=question,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    found_expected_file = False

    print()
    print("TOP RESULTS:")

    for i in range(len(documents)):

        metadata = metadatas[i]

        file_path = metadata["file_path"]
        chunk_name = metadata["name"]
        distance = distances[i]

        print()
        print(f"{i + 1}. {file_path}")
        print(f"   Chunk: {chunk_name}")
        print(f"   Distance: {distance:.4f}")

        if expected_file.lower() in file_path.lower():
            found_expected_file = True

    # --------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------

    print()

    if found_expected_file:
        print("STATUS: PASS")
        print(f"Expected file '{expected_file}' found in top 5 results.")

    else:
        print("STATUS: FAIL")
        print(f"Expected file '{expected_file}' NOT found in top 5 results.")


print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)