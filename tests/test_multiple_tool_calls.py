import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.agent.nodes import CodeToolCallingAgent


def main():

    print("=" * 80)
    print("MULTIPLE TOOL CALLS TEST")
    print("=" * 80)

    print()
    print("Initializing agent...")

    agent = CodeToolCallingAgent()

    # --------------------------------------------------
    # TEST 1 — Execute multiple search calls directly
    # --------------------------------------------------

    print()
    print("TEST 1 — Multiple tool calls")

    tool_calls = [
        {
            "function": {
                "name": "code_search",
                "arguments": {
                    "query": (
                        "Which code extracts classes and "
                        "functions from Python files?"
                    ),
                    "top_k": 3,
                },
            },
        },
        {
            "function": {
                "name": "code_search",
                "arguments": {
                    "query": (
                        "Which code creates embeddings "
                        "for code chunks?"
                    ),
                    "top_k": 3,
                },
            },
        },
    ]

    results = agent.execute_tool_calls(
        tool_calls
    )

    if len(results) != 2:

        raise AssertionError(
            "Expected exactly 2 tool results."
        )

    for index, result in enumerate(
        results,
        start=1
    ):

        if result["role"] != "tool":

            raise AssertionError(
                f"Tool result {index} has an invalid role."
            )

        if not result["content"]:

            raise AssertionError(
                f"Tool result {index} is empty."
            )

        print(
            f"Tool call {index}: PASS"
        )

    print("TEST 1: PASS")

    # --------------------------------------------------
    # TEST 2 — Unknown tool
    # --------------------------------------------------

    print()
    print("TEST 2 — Unknown tool rejection")

    invalid_tool_calls = [
        {
            "function": {
                "name": "unknown_tool",
                "arguments": {},
            },
        }
    ]

    try:

        agent.execute_tool_calls(
            invalid_tool_calls
        )

        raise AssertionError(
            "Unknown tool was not rejected."
        )

    except ValueError as error:

        print(
            f"Correctly rejected: {error}"
        )

    print("TEST 2: PASS")

    # --------------------------------------------------
    # TEST 3 — Real LLM tool-calling flow
    # --------------------------------------------------

    print()
    print("TEST 3 — Real LLM tool-calling flow")

    question = (
        "Find the code responsible for extracting "
        "classes and functions from Python files, "
        "and explain the implementation."
    )

    answer = agent.ask(
        question=question
    )

    if not answer.strip():

        raise AssertionError(
            "LLM returned an empty answer."
        )

    print()
    print("Final answer:")
    print("-" * 80)
    print(answer)
    print("-" * 80)

    answer_lower = answer.lower()

    expected_terms = [
        "codebasechunker",
        "chunk_file",
    ]

    for term in expected_terms:

        if term not in answer_lower:

            raise AssertionError(
                f"Expected term not found: {term}"
            )

    print()
    print("TEST 3: PASS")

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()