import sys
import os
import json

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.agent.nodes import CodeToolCallingAgent


def main():

    print("=" * 80)
    print("TOOL ERROR HANDLING TEST")
    print("=" * 80)

    print()
    print("Initializing agent...")

    agent = CodeToolCallingAgent()

    # --------------------------------------------------
    # TEST 1 — Valid tool call
    # --------------------------------------------------

    print()
    print("TEST 1 — Valid tool call")

    result = agent.execute_tool(
        tool_name="code_search",
        arguments={
            "query": (
                "Which code extracts classes and "
                "functions?"
            ),
            "top_k": 3,
        },
    )

    parsed_result = json.loads(result)

    if parsed_result["success"] is not True:

        raise AssertionError(
            "Valid tool call was reported as a failure."
        )

    if not parsed_result["results"]:

        raise AssertionError(
            "Valid tool call returned no results."
        )

    print("Valid tool call returned results.")
    print("TEST 1: PASS")

    # --------------------------------------------------
    # TEST 2 — Empty query
    # --------------------------------------------------

    print()
    print("TEST 2 — Empty query error")

    result = agent.execute_tool(
        tool_name="code_search",
        arguments={
            "query": "",
            "top_k": 5,
        },
    )

    parsed_result = json.loads(result)

    if parsed_result["success"] is not False:

        raise AssertionError(
            "Empty query was not reported as a failure."
        )

    if parsed_result["error"] != "ValueError":

        raise AssertionError(
            "Unexpected error type for empty query: "
            f"{parsed_result['error']}"
        )

    print(
        f"Correctly returned error: "
        f"{parsed_result['message']}"
    )

    print("TEST 2: PASS")

    # --------------------------------------------------
    # TEST 3 — Unknown tool
    # --------------------------------------------------

    print()
    print("TEST 3 — Unknown tool error")

    result = agent.execute_tool(
        tool_name="unknown_tool",
        arguments={}
    )

    parsed_result = json.loads(result)

    if parsed_result["success"] is not False:

        raise AssertionError(
            "Unknown tool was not reported as a failure."
        )

    if parsed_result["error"] != "ValueError":

        raise AssertionError(
            "Unexpected error type for unknown tool: "
            f"{parsed_result['error']}"
        )

    print(
        f"Correctly returned error: "
        f"{parsed_result['message']}"
    )

    print("TEST 3: PASS")

    # --------------------------------------------------
    # TEST 4 — Missing query argument
    # --------------------------------------------------

    print()
    print("TEST 4 — Missing query argument")

    result = agent.execute_tool(
        tool_name="code_search",
        arguments={}
    )

    parsed_result = json.loads(result)

    if parsed_result["success"] is not False:

        raise AssertionError(
            "Missing query was not reported as a failure."
        )

    if parsed_result["error"] not in [
        "KeyError",
        "ValueError",
        "TypeError",
    ]:

        raise AssertionError(
            "Unexpected error type for missing query: "
            f"{parsed_result['error']}"
        )

    print(
        f"Correctly returned error: "
        f"{parsed_result['message']}"
    )

    print("TEST 4: PASS")

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()