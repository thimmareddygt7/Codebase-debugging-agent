from langgraph.graph import StateGraph, START, END
from src.agent.memory import AgentState, SessionManager
from src.agent.nodes import (
    classify_intent_node,
    llm_node,
    tool_node,
    cross_check_context_node,
    validate_fix_node,
    finalize_node,
    should_continue_tools,
    should_retry_or_finish,
)

session_manager = SessionManager(db_path="data/sessions.db")


def create_agent_graph():
    """Builds the state machine with pre-synthesis context cross-checking."""
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("cross_check_context", cross_check_context_node)
    workflow.add_node("validate_fix", validate_fix_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "llm")

    # If LLM decides to call tools, route to tools; otherwise go to fix validation
    workflow.add_conditional_edges(
        "llm",
        should_continue_tools,
        {
            "tools": "tools",
            "validate_fix": "validate_fix",
        }
    )

    # NEW: Tools output flows through context cross-checking before re-invoking LLM
    workflow.add_edge("tools", "cross_check_context")
    workflow.add_edge("cross_check_context", "llm")

    workflow.add_conditional_edges(
        "validate_fix",
        should_retry_or_finish,
        {
            "retry_llm": "llm",
            "finalize": "finalize",
        }
    )

    workflow.add_edge("finalize", END)

    checkpointer = session_manager.get_checkpointer()
    return workflow.compile(checkpointer=checkpointer)


agent_graph = create_agent_graph()


def ask_codebase_agent(
    question: str,
    thread_id: str = "default_session",
    max_retries: int = 2
) -> dict:
    """Runs a question through the verified debugging agent state machine."""
    config = session_manager.get_thread_config(thread_id)
    current_snapshot = agent_graph.get_state(config)

    base_system_prompt = (
        "You are an expert AI Codebase Debugging Agent. "
        "Ground all answers strictly on retrieved code context. "
        "When fixing bugs or proposing code changes, provide valid, clean, and executable "
        "Python code wrapped in ```python ... ``` markdown blocks. "
        "Always use your tools ('code_search', 'doc_search') before answering."
    )

    if not current_snapshot.values:
        initial_input: AgentState = {
            "question": question,
            "query_type": None,
            "messages": [
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": question}
            ],
            "tool_calls": [],
            "tool_results": [],
            "final_answer": "",
            "error": None,
            "step_count": 0,
            "retry_count": 0,
            "max_retries": max_retries,
            "fix_valid": True,
            "validation_feedback": None,
            "investigated_files": [],
            "active_symbols": [],
            "debug_notes": [],
            "context_verified": False,
            "context_confidence": 0.0,
            "verification_notes": None,
        }
    else:
        prior_state = current_snapshot.values
        investigated_files = prior_state.get("investigated_files", [])
        active_symbols = prior_state.get("active_symbols", [])

        context_hint = ""
        if investigated_files or active_symbols:
            context_hint = (
                f"\n[Active Debugging Context: Investigated Files: {investigated_files}, "
                f"Active Symbols: {active_symbols}]\n"
            )

        existing_messages = list(prior_state["messages"])
        existing_messages.append({"role": "user", "content": context_hint + question})

        initial_input: AgentState = {
            **prior_state,
            "question": question,
            "messages": existing_messages,
            "tool_calls": [],
            "tool_results": [],
            "final_answer": "",
            "error": None,
            "step_count": 0,
            "retry_count": 0,
            "max_retries": max_retries,
            "fix_valid": True,
            "validation_feedback": None,
            "context_verified": False,
            "context_confidence": 0.0,
            "verification_notes": None,
        }

    return agent_graph.invoke(initial_input, config=config)