"""
Conditional workflow branching and routing logic for the Codebase Debugging Agent.
"""
from typing import Literal
from src.agent.memory import AgentState

MAX_WORKFLOW_STEPS = 6


def should_continue_tools(state: AgentState) -> Literal["tools", "validate_fix"]:
    """
    Evaluates whether the agent state indicates pending tool execution calls
    or should transition directly to code validation.
    """
    if state.get("step_count", 0) >= state.get("max_steps", MAX_WORKFLOW_STEPS):
        return "validate_fix"

    messages = state.get("messages", [])
    if not messages:
        return "validate_fix"

    latest_message = messages[-1]
    if latest_message.get("tool_calls"):
        return "tools"

    return "validate_fix"


def route_by_error_type(error_type: str) -> Literal["syntax_fix", "runtime_debug", "logic_investigation"]:
    """
    Routes debugging workflow based on classified error taxonomy.
    """
    error_lower = (error_type or "").lower()
    if any(keyword in error_lower for keyword in ["syntax", "indentation", "parse", "token"]):
        return "syntax_fix"
    elif any(keyword in error_lower for keyword in ["runtime", "exception", "null", "typeerror", "attributeerror"]):
        return "runtime_debug"
    return "logic_investigation"
