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
    print("CODE SEARCH TOOL VALIDATION TEST")
    print("=" * 80)

    print()
    print("Initializing code search tool...")

    tool = CodeSearchTool()

    # --------------------------------------------------
    # TEST 1 — Valid search
    # --------------------------------------------------

    print()
    print("TEST 1 — Valid search")

    results = tool.search(
        query="Which code extracts classes and functions?",
        top_k=5
    )

    if not results:

        raise AssertionError(
            "Valid search returned no results."
        )

    print(
        f"Returned {len(results)} valid results."
    )

    for result in results:

        tool.validate_result(result)

    print("PASS")

    # --------------------------------------------------
    # TEST 2 — Empty query
    # --------------------------------------------------

    print()
    print("TEST 2 — Empty query")

    try:

        tool.search(
            query="",
            top_k=5
        )

        raise AssertionError(
            "Empty query was not rejected."
        )

    except ValueError as error:

        print(
            f"Correctly rejected: {error}"
        )

    print("PASS")

    # --------------------------------------------------
    # TEST 3 — Invalid top_k
    # --------------------------------------------------

    print()
    print("TEST 3 — Invalid top_k")

    try:

        tool.search(
            query="find classes",
            top_k=0
        )

        raise AssertionError(
            "Invalid top_k was not rejected."
        )

    except ValueError as error:

        print(
            f"Correctly rejected: {error}"
        )

    print("PASS")

    # --------------------------------------------------
    # TEST 4 — Invalid query type
    # --------------------------------------------------

    print()
    print("TEST 4 — Invalid query type")

    try:

        tool.search(
            query=123,
            top_k=5
        )

        raise AssertionError(
            "Invalid query type was not rejected."
        )

    except TypeError as error:

        print(
            f"Correctly rejected: {error}"
        )

    print("PASS")

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()