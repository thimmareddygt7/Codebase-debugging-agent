"""
Iterative retry, self-correction loop, and backoff control for the Codebase Debugging Agent.
"""
from typing import Literal
from src.agent.memory import AgentState

DEFAULT_MAX_RETRIES = 3


def should_retry_or_finish(state: AgentState) -> Literal["retry_llm", "finalize"]:
    """
    Decides whether to trigger an automated self-correction loop back to the LLM
    with diagnostic AST error feedback, or finalize and checkpoint the state.
    """
    if state.get("fix_valid", True):
        return "finalize"

    current_retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if current_retries < max_retries:
        return "retry_llm"

    return "finalize"


def is_retry_budget_exhausted(retry_count: int, max_retries: int = DEFAULT_MAX_RETRIES) -> bool:
    """Checks whether the iterative retry loop has exhausted permitted attempts."""
    return retry_count >= max_retries
