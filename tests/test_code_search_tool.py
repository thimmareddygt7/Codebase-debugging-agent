import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.agent.tools import CodeSearchTool


def main():

    print("=" * 80)
    print("CODE SEARCH TOOL TEST")
    print("=" * 80)

    print()
    print("Initializing code search tool...")

    tool = CodeSearchTool()

    print()
    print("Searching codebase...")

    query = (
        "Which code extracts classes and functions "
        "from Python files?"
    )

    results = tool.search(
        query=query,
        top_k=5
    )

    print()
    print("Query:")
    print(query)

    print()
    print("Retrieved results:")
    print("-" * 80)

    for index, result in enumerate(results, start=1):

        print()
        print(f"RESULT {index}")
        print(f"File       : {result['file_path']}")
        print(f"Chunk      : {result['chunk_name']}")
        print(f"Qualified  : {result['qualified_name']}")
        print(
            f"Lines      : "
            f"{result['start_line']}-{result['end_line']}"
        )
        print(f"Distance   : {result['distance']:.4f}")

        print()
        print("Code:")
        print(result["code"])

        print("-" * 80)

    if not results:
        raise AssertionError(
            "Code search tool returned no results."
        )

    expected_file = "src\\ingestion\\chunker.py"

    found_expected_file = any(
        result["file_path"] == expected_file
        for result in results
    )

    if not found_expected_file:
        raise AssertionError(
            "Expected file was not found in search results: "
            f"{expected_file}"
        )

    print()
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()