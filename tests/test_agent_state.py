import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.agent.memory import AgentState
from src.agent.nodes import CodeToolCallingAgent


def main():

    print("=" * 80)
    print("AGENT STATE TEST")
    print("=" * 80)

    # --------------------------------------------------
    # TEST 1 — Create state
    # --------------------------------------------------

    print()
    print("TEST 1 — Create agent state")

    state = AgentState(
        question="Find the code that extracts classes."
    )

    if state.question != (
        "Find the code that extracts classes."
    ):

        raise AssertionError(
            "Question was not stored correctly."
        )

    if state.messages != []:

        raise AssertionError(
            "Messages should initially be empty."
        )

    if state.tool_calls != []:

        raise AssertionError(
            "Tool calls should initially be empty."
        )

    if state.tool_results != []:

        raise AssertionError(
            "Tool results should initially be empty."
        )

    if state.final_answer != "":

        raise AssertionError(
            "Final answer should initially be empty."
        )

    print("State created correctly.")
    print("TEST 1: PASS")

    # --------------------------------------------------
    # TEST 2 — State updates
    # --------------------------------------------------

    print()
    print("TEST 2 — State updates")

    state.add_message(
        {
            "role": "user",
            "content": "Find classes.",
        }
    )

    state.add_tool_call(
        {
            "function": {
                "name": "code_search"
            }
        }
    )

    state.add_tool_result(
        {
            "tool_name": "code_search",
            "result": "test result",
        }
    )

    state.set_final_answer(
        "Found the code."
    )

    if len(state.messages) != 1:

        raise AssertionError(
            "Message was not stored."
        )

    if len(state.tool_calls) != 1:

        raise AssertionError(
            "Tool call was not stored."
        )

    if len(state.tool_results) != 1:

        raise AssertionError(
            "Tool result was not stored."
        )

    if state.final_answer != "Found the code.":

        raise AssertionError(
            "Final answer was not stored."
        )

    print("State updates correctly.")
    print("TEST 2: PASS")

    # --------------------------------------------------
    # TEST 3 — Real agent state flow
    # --------------------------------------------------

    print()
    print("TEST 3 — Real agent state flow")

    agent = CodeToolCallingAgent()

    question = (
        "Which code extracts classes and functions "
        "from Python files?"
    )

    answer = agent.ask(
        question=question
    )

    if not answer.strip():

        raise AssertionError(
            "Agent returned an empty answer."
        )

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

    print("Agent returned a valid answer.")
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