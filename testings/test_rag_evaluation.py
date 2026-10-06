import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.retriever import CodeRetriever


retriever = CodeRetriever()


evaluation_queries = [
    {
        "question": "Which code extracts classes and functions from Python files?",
        "expected_file": "src\\ingestion\\chunker.py",
    },
    {
        "question": "How are source files loaded from the codebase?",
        "expected_file": "src\\ingestion\\loader.py",
    },
    {
        "question": "How are code chunks converted into embeddings?",
        "expected_file": "src\\ingestion\\embedder.py",
    },
    {
        "question": "How are embeddings stored and searched?",
        "expected_file": "src\\retrieval\\vector_store.py",
    },
]


print()
print("=" * 80)
print("RAG RETRIEVAL EVALUATION")
print("=" * 80)


passed = 0
failed = 0


for index, item in enumerate(evaluation_queries, start=1):

    question = item["question"]
    expected_file = item["expected_file"]

    print()
    print("-" * 80)
    print(f"TEST {index}")
    print("-" * 80)

    print()
    print("Question:")
    print(question)

    results = retriever.search(
        query=question,
        top_k=5
    )

    metadatas = results["metadatas"][0]

    retrieved_files = [
        metadata["file_path"]
        for metadata in metadatas
    ]

    print()
    print("Expected file:")
    print(expected_file)

    print()
    print("Retrieved files:")

    for file_path in retrieved_files:
        print(f"  - {file_path}")

    if expected_file in retrieved_files:

        print()
        print("RESULT: PASS")

        passed += 1

    else:

        print()
        print("RESULT: FAIL")

        failed += 1


total = len(evaluation_queries)

accuracy = passed / total


print()
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print()
print(f"Total tests : {total}")
print(f"Passed      : {passed}")
print(f"Failed      : {failed}")
print(f"Accuracy    : {accuracy:.2%}")


assert passed > 0


print()
print("=" * 80)
print("STATUS: PASS")
print("=" * 80)