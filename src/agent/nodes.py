import json
import ollama
from typing import Dict, Any, List
from src.agent.tools import CodeSearchTool, DocSearchTool
from src.agent.memory import AgentState
from src.agent.validator import FixValidator
from src.agent.context_verifier import ContextCrossChecker

# Initialize components
code_search_tool = CodeSearchTool()
doc_search_tool = DocSearchTool()
fix_validator = FixValidator()
context_verifier = ContextCrossChecker()

tools_registry = {
    CodeSearchTool.NAME: code_search_tool,
    DocSearchTool.NAME: doc_search_tool,
}

tools_schema = [
    code_search_tool.get_tool_definition(),
    doc_search_tool.get_tool_definition(),
]

DEFAULT_MODEL = "qwen3:4b"
MAX_STEPS = 5


def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes user question or error to classify search branch."""
    question = state["question"].lower()
    doc_keywords = ["how to run", "setup", "readme", "config", "environment", "install", "docker", "requirements", "license"]
    code_keywords = ["error", "exception", "traceback", "def ", "class ", "function", "method", "where is", "syntax", "line", "fix"]

    if any(k in question for k in doc_keywords):
        query_type = "doc_query"
    elif any(k in question for k in code_keywords):
        query_type = "code_query"
    else:
        query_type = "general"

    return {"query_type": query_type}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Dispatches execution to the corresponding tool in the registry."""
    try:
        tool_instance = tools_registry.get(tool_name)
        if not tool_instance:
            raise ValueError(f"Unknown tool: {tool_name}")

        if tool_name == CodeSearchTool.NAME:
            results = tool_instance.execute(
                query=arguments["query"],
                top_k=arguments.get("top_k", 5)
            )
        elif tool_name == DocSearchTool.NAME:
            results = tool_instance.execute(
                query=arguments["query"]
            )
        else:
            results = []

        return json.dumps(
            {"success": True, "tool_used": tool_name, "results": results},
            indent=2,
            default=str
        )
    except Exception as error:
        return json.dumps(
            {
                "success": False,
                "error": type(error).__name__,
                "message": str(error)
            },
            indent=2
        )


def _serialize_message(raw_msg: Any) -> Dict[str, Any]:
    """Ensures Ollama objects are converted to standard Python dicts for SQLite serialization."""
    role = getattr(raw_msg, "role", None) or (raw_msg.get("role") if isinstance(raw_msg, dict) else "assistant")
    content = getattr(raw_msg, "content", None) or (raw_msg.get("content") if isinstance(raw_msg, dict) else "")

    serialized: Dict[str, Any] = {
        "role": str(role),
        "content": str(content)
    }

    raw_tool_calls = getattr(raw_msg, "tool_calls", None) or (raw_msg.get("tool_calls") if isinstance(raw_msg, dict) else None)
    if raw_tool_calls:
        clean_calls = []
        for tc in raw_tool_calls:
            if hasattr(tc, "model_dump"):
                clean_calls.append(tc.model_dump())
            elif isinstance(tc, dict):
                clean_calls.append(dict(tc))
            else:
                clean_calls.append({
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", ""),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", {}),
                    }
                })
        serialized["tool_calls"] = clean_calls

    return serialized


def llm_node(state: AgentState) -> Dict[str, Any]:
    """Calls LLM with tools and serializes messages cleanly."""
    messages = list(state["messages"])

    response = ollama.chat(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=tools_schema,
    )

    clean_message = _serialize_message(response["message"])
    new_messages = list(messages)
    new_messages.append(clean_message)

    return {"messages": new_messages}


def tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes tool calls and accumulates investigated artifacts."""
    latest_message = state["messages"][-1]
    tool_calls = latest_message.get("tool_calls", [])

    new_tool_calls = list(state.get("tool_calls", []))
    new_tool_results = list(state.get("tool_results", []))
    new_messages = list(state["messages"])

    investigated_files = list(state.get("investigated_files", []))
    active_symbols = list(state.get("active_symbols", []))

    for tool_call in tool_calls:
        function = tool_call["function"]
        tool_name = function["name"]
        arguments = function.get("arguments", {})

        new_tool_calls.append(tool_call)
        raw_result_str = execute_tool(tool_name=tool_name, arguments=arguments)

        try:
            parsed = json.loads(raw_result_str)
            if parsed.get("success"):
                for item in parsed.get("results", []):
                    fp = item.get("file_path")
                    if fp and fp not in investigated_files:
                        investigated_files.append(fp)

                    sym = item.get("qualified_name") or item.get("chunk_name")
                    if sym and sym not in active_symbols:
                        active_symbols.append(sym)
        except Exception:
            pass

        new_tool_results.append({
            "tool_name": tool_name,
            "result": raw_result_str
        })

        new_messages.append({
            "role": "tool",
            "content": raw_result_str
        })

    current_steps = state.get("step_count", 0) + 1

    return {
        "messages": new_messages,
        "tool_calls": new_tool_calls,
        "tool_results": new_tool_results,
        "step_count": current_steps,
        "investigated_files": investigated_files,
        "active_symbols": active_symbols,
    }


# --------------------------------------------------
# NEW: Context Cross-Checking Node (v7a)
# --------------------------------------------------
def cross_check_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Cross-checks retrieved snippets before answering to prevent hallucinations.
    """
    question = state["question"]
    tool_results = state.get("tool_results", [])
    snippets = context_verifier.extract_search_documents(tool_results)

    is_verified, score, explanation = context_verifier.verify_relevance(question, snippets)

    new_messages = list(state["messages"])
    if not is_verified:
        # Warn LLM that retrieved snippets did not match well, enforcing strict fidelity
        guard_prompt = (
            f"[Context Cross-Check Warning: Relevance score was low ({score:.2f}). "
            f"Note: {explanation}. If the retrieved code does not clearly answer the user's question, "
            f"state clearly that the codebase does not provide this information rather than guessing.]"
        )
        new_messages.append({"role": "system", "content": guard_prompt})

    return {
        "context_verified": is_verified,
        "context_confidence": score,
        "verification_notes": explanation,
        "messages": new_messages,
    }


def validate_fix_node(state: AgentState) -> Dict[str, Any]:
    """Validates the proposed fix or code blocks generated by the LLM."""
    latest_message = state["messages"][-1]
    content = latest_message.get("content", "")

    is_valid, feedback = fix_validator.evaluate_proposed_fix(content)

    new_messages = list(state["messages"])
    current_retries = state.get("retry_count", 0)

    if not is_valid:
        current_retries += 1
        correction_prompt = (
            f"Validation Failed: {feedback}\n"
            "Please regenerate your suggested fix addressing this exact error. "
            "Ensure all Python code inside markdown blocks is valid, executable syntax."
        )
        new_messages.append({
            "role": "user",
            "content": correction_prompt
        })

    return {
        "fix_valid": is_valid,
        "validation_feedback": feedback,
        "retry_count": current_retries,
        "messages": new_messages,
    }


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Extracts final answer and updates session debug notes."""
    if not state["messages"]:
        return {"error": "No messages available.", "final_answer": ""}

    final_answer = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_answer = msg["content"].strip()
            break

    if not final_answer:
        return {"error": "No assistant response generated.", "final_answer": ""}

    debug_notes = list(state.get("debug_notes", []))
    summary_entry = f"Q: {state['question'][:60]}... -> Answered (Verified: {state.get('context_verified', False)})"
    if summary_entry not in debug_notes:
        debug_notes.append(summary_entry)

    return {
        "final_answer": final_answer,
        "error": None,
        "debug_notes": debug_notes,
    }


def should_continue_tools(state: AgentState) -> str:
    """Routes to tools if tool calls exist, else routes to validation."""
    if state.get("step_count", 0) >= MAX_STEPS:
        return "validate_fix"

    if not state.get("messages"):
        return "validate_fix"

    latest_message = state["messages"][-1]
    if latest_message.get("tool_calls"):
        return "tools"

    return "validate_fix"


def should_retry_or_finish(state: AgentState) -> str:
    """Decides whether to retry the LLM with error feedback or finalize."""
    if state.get("fix_valid", True):
        return "finalize"

    if state.get("retry_count", 0) < state.get("max_retries", 3):
        return "retry_llm"

    return "finalize"