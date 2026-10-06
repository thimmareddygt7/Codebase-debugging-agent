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
    print("V3.7 — AGENT NODE TEST")
    print("=" * 80)

    agent = CodeToolCallingAgent()

    # --------------------------------------------------
    # TEST 1 — State initialization
    # --------------------------------------------------

    print()
    print("TEST 1 — State initialization")

    question = (
        "Which code extracts classes and functions "
        "from Python files?"
    )

    state = agent.initialize_state(
        question=question
    )

    if state.question != question:

        raise AssertionError(
            "Question was not stored correctly."
        )

    if len(state.messages) != 2:

        raise AssertionError(
            "Expected system and user messages."
        )

    print("State initialized correctly.")
    print("TEST 1: PASS")

    # --------------------------------------------------
    # TEST 2 — LLM node
    # --------------------------------------------------

    print()
    print("TEST 2 — LLM node")

    state = agent.llm_node(
        state
    )

    if len(state.messages) != 3:

        raise AssertionError(
            "LLM response was not added to state."
        )

    latest_message = state.messages[-1]

    if "content" not in latest_message:

        raise AssertionError(
            "LLM response does not contain content."
        )

    print("LLM node updated the state.")
    print("TEST 2: PASS")

    # --------------------------------------------------
    # TEST 3 — Tool routing detection
    # --------------------------------------------------

    print()
    print("TEST 3 — Tool routing")

    if not agent.has_tool_calls(
        state
    ):

        raise AssertionError(
            "Expected the LLM to request a tool."
        )

    print("Tool call detected.")
    print("TEST 3: PASS")

    # --------------------------------------------------
    # TEST 4 — Tool node
    # --------------------------------------------------

    print()
    print("TEST 4 — Tool node")

    state = agent.tool_node(
        state
    )

    if len(state.tool_calls) == 0:

        raise AssertionError(
            "Tool call was not recorded."
        )

    if len(state.tool_results) == 0:

        raise AssertionError(
            "Tool result was not recorded."
        )

    if len(state.messages) < 4:

        raise AssertionError(
            "Tool message was not added to state."
        )

    print("Tool node updated the state.")
    print("TEST 4: PASS")

    # --------------------------------------------------
    # TEST 5 — LLM after tool
    # --------------------------------------------------

    print()
    print("TEST 5 — LLM after tool")

    state = agent.llm_node(
        state
    )

    if len(state.messages) < 5:

        raise AssertionError(
            "Second LLM response was not added."
        )

    if agent.has_tool_calls(
        state
    ):

        raise AssertionError(
            "Expected final answer after tool execution."
        )

    print("LLM produced the final response.")
    print("TEST 5: PASS")

    # --------------------------------------------------
    # TEST 6 — Final answer
    # --------------------------------------------------

    print()
    print("TEST 6 — Final answer")

    state = agent.set_final_answer(
        state
    )

    if not state.final_answer.strip():

        raise AssertionError(
            "Final answer was empty."
        )

    answer_lower = (
        state.final_answer.lower()
    )

    expected_terms = [
        "codebasechunker",
    ]

    for term in expected_terms:

        if term not in answer_lower:

            raise AssertionError(
                f"Expected term not found: {term}"
            )

    print("Final answer stored correctly.")
    print("TEST 6: PASS")

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()