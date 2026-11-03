import sys
import os

# Add the project root to Python's import path.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.rag.chain import CodeRAGChain


def main():
    rag = CodeRAGChain()

    evaluation_queries = [
        {
            "question": (
                "Which code extracts classes and functions "
                "from Python files?"
            ),
            "expected_terms": [
                "chunk_file",
                "_traverse",
                "CodebaseChunker",
            ],
        },
        {
            "question": "How does the system load source files?",
            "expected_terms": [
                "CodebaseLoader",
                "get_source_files",
            ],
        },
        {
            "question": (
                "How are code chunks converted into embeddings?"
            ),
            "expected_terms": [
                "CodeEmbedder",
                "embed_texts",
            ],
        },
        {
            "question": "How are embeddings stored and searched?",
            "expected_terms": [
                "CodeVectorStore",
                "add_chunks",
                "search",
            ],
        },
    ]

    print()
    print("=" * 80)
    print("RAG ANSWER EVALUATION")
    print("=" * 80)

    passed = 0
    failed = 0

    total = len(evaluation_queries)

    for index, item in enumerate(evaluation_queries, start=1):

        question = item["question"]
        expected_terms = item["expected_terms"]

        print()
        print("-" * 80)
        print(f"TEST {index}/{total}")
        print("-" * 80)

        print()
        print("Question:")
        print(question)

        try:
            result = rag.ask(
                question=question,
                top_k=5,
            )

            answer = result["answer"]
            citations = result["citations"]

            print()
            print("Answer:")
            print(answer)

            print()
            print("Sources:")

            for citation in citations:
                print(citation)

            # Check that the answer contains every expected term.
            answer_lower = answer.lower()

            missing_terms = [
                term
                for term in expected_terms
                if term.lower() not in answer_lower
            ]

            if not answer.strip():
                print()
                print("RESULT: FAIL")
                print("Reason: The generated answer is empty.")
                failed += 1

            elif missing_terms:
                print()
                print("RESULT: FAIL")
                print("Missing expected terms:")

                for term in missing_terms:
                    print(f"  - {term}")

                failed += 1

            else:
                print()
                print("RESULT: PASS")
                passed += 1

        except Exception as error:
            print()
            print("RESULT: FAIL")
            print(f"Error: {error}")
            failed += 1

    accuracy = (passed / total) * 100 if total else 0.0

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print()
    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Accuracy    : {accuracy:.2f}%")

    print()

    if failed == 0:
        print("=" * 80)
        print("STATUS: PASS")
        print("=" * 80)
    else:
        print("=" * 80)
        print("STATUS: FAIL")
        print("=" * 80)

        raise AssertionError(
            f"RAG answer evaluation failed: "
            f"{failed} out of {total} test(s) failed."
        )


if __name__ == "__main__":
    main()