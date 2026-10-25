import json
import os
import groq
from typing import Dict, Any, List
from src.config import settings
from src.observability import trace_run
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

MAX_STEPS = 5


@trace_run(name="[Phase 1: START] Intent Classification", run_type="chain", tags=["phase-1", "classification"])
def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes user question or error to classify search branch."""
    question = state["question"]
    print("\n" + "=" * 80)
    print(f"[STEP 1/6] >>> PHASE 1: START - Intent Classification & Route Selection")
    print("=" * 80)
    print(f"    -> Input Question: '{question[:90]}...'")

    q_lower = question.lower()
    doc_keywords = ["how to run", "setup", "readme", "config", "environment", "install", "docker", "requirements", "license"]
    code_keywords = ["error", "exception", "traceback", "def ", "class ", "function", "method", "where is", "syntax", "line", "fix"]

    if any(k in q_lower for k in doc_keywords):
        query_type = "doc_query"
        print(f"    -> Classification: 'doc_query' (Documentation & environment setup)")
    elif any(k in q_lower for k in code_keywords):
        query_type = "code_query"
        print(f"    -> Classification: 'code_query' (Codebase AST structure & bug debugging)")
    else:
        query_type = "general"
        print(f"    -> Classification: 'general' (General architectural inquiry)")

    print(f"    -> Next Destination: PHASE 2 (LLM Reasoner)")
    return {"query_type": query_type}


@trace_run(name="tool_execution", run_type="tool")
def execute_tool(tool_name: str, arguments: dict) -> str:
    """Dispatches execution to the corresponding tool in the registry."""
    print(f"    -> Calling Tool '{tool_name}' with arguments: {arguments}")
    try:
        tool_instance = tools_registry.get(tool_name)
        if not tool_instance:
            raise ValueError(f"Unknown tool: '{tool_name}' is not registered in tools_registry")

        if tool_name == CodeSearchTool.NAME:
            top_k = min(int(arguments.get("top_k", 3)), 3)
            results = tool_instance.execute(
                query=arguments["query"],
                top_k=top_k
            )
            # Trim code snippets to keep token usage within Groq free limits
            for item in results:
                if "code" in item and len(item["code"]) > 750:
                    item["code"] = item["code"][:750] + "\n... [truncated for token efficiency]"
            print(f"       <- Tool Execution Succeeded: Retrieved {len(results)} relevant code chunk(s) (top_k={top_k})")
        elif tool_name == DocSearchTool.NAME:
            results = tool_instance.execute(
                query=arguments["query"]
            )
            print(f"       <- Tool Execution Succeeded: Retrieved {len(results)} relevant doc match(es)")
        else:
            results = []

        return json.dumps(
            {"success": True, "tool_used": tool_name, "results": results},
            indent=2,
            default=str
        )
    except Exception as error:
        print("\n" + "*" * 80)
        print(f"[ERROR IN PHASE 3: TOOL EXECUTION] Tool '{tool_name}' encountered an error:")
        print(f"    Error Type: {type(error).__name__}")
        print(f"    Details:    {error}")
        print("    Recovery:   Returning error envelope to LLM to gracefully handle missing tool data.")
        print("*" * 80 + "\n")
        return json.dumps(
            {
                "success": False,
                "error": type(error).__name__,
                "message": str(error)
            },
            indent=2
        )


def _serialize_message(raw_msg: Any) -> Dict[str, Any]:
    """Serializes message object to standard Python dict for SQLite serialization."""
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
            call_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "call_func")
            if hasattr(tc, "function"):
                clean_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": getattr(tc.function, "name", ""),
                        "arguments": getattr(tc.function, "arguments", {}),
                    }
                })
            elif isinstance(tc, dict):
                clean_calls.append(dict(tc))
        serialized["tool_calls"] = clean_calls

    return serialized


@trace_run(name="groq_completion", run_type="llm", tags=["groq"])
def _call_groq(messages: List[Dict[str, Any]], allow_tools: bool = True) -> Dict[str, Any]:
    """Executes tool-calling completion via Groq with rate-limit fallback and OTPM guard."""
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your .env file or environment variable."
        )

    client = groq.Groq(api_key=api_key)

    # Context window protection: keep system prompt + recent messages to avoid exceeding free limits
    if len(messages) > 6:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent = messages[-5:]
        while recent and recent[0].get("role") == "tool":
            recent = recent[1:]
        messages = (system_msgs[:1] if system_msgs else []) + recent

    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        item: Dict[str, Any] = {"role": role, "content": content}

        if role == "assistant" and msg.get("tool_calls"):
            groq_tool_calls = []
            for tc in msg["tool_calls"]:
                call_id = tc.get("id") or "call_default"
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                groq_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args
                    }
                })
            item["tool_calls"] = groq_tool_calls

        elif role == "tool":
            item["tool_call_id"] = msg.get("tool_call_id") or "call_default"
            if "name" in msg:
                item["name"] = msg["name"]

        formatted_messages.append(item)

    model = settings.GROQ_MODEL
    tools_param = tools_schema if allow_tools else None
    tool_choice_param = "auto" if allow_tools else None

    print(f"       -> [GROQ DISPATCH] Sending request to Groq Cloud API...")
    print(f"          Model: {model} | Context Messages: {len(formatted_messages)} | Tools Enabled: {allow_tools} | max_tokens: 800")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            tools=tools_param,
            tool_choice=tool_choice_param,
            max_tokens=800,
        )
    except (groq.RateLimitError, groq.APIStatusError) as exc:
        is_quota_issue = (
            isinstance(exc, groq.RateLimitError) or
            getattr(exc, "status_code", None) in (413, 429) or
            "rate_limit_exceeded" in str(exc).lower() or
            "request too large" in str(exc).lower()
        )
        print("\n" + "*" * 80)
        print(f"[ERROR IN PHASE 2: LLM API CALL]")
        print(f"    Error: {type(exc).__name__}: {exc}")
        if is_quota_issue and settings.FALLBACK_MODEL and settings.FALLBACK_MODEL != model:
            print(f"    -> AUTO-RECOVERY: Switching to Fallback Engine: '{settings.FALLBACK_MODEL}'")
            print("*" * 80 + "\n")
            completion = client.chat.completions.create(
                model=settings.FALLBACK_MODEL,
                messages=formatted_messages,
                tools=tools_param,
                tool_choice=tool_choice_param,
                max_tokens=800,
            )
        else:
            print("*" * 80 + "\n")
            raise

    usage = getattr(completion, "usage", None)
    if usage:
        print(f"       <- [GROQ RESPONSE RECEIVED] Prompt Tokens: {usage.prompt_tokens} | Completion Tokens: {usage.completion_tokens} | Total: {usage.total_tokens}")
    else:
        print(f"       <- [GROQ RESPONSE RECEIVED] (Success)")

    choice_msg = completion.choices[0].message
    serialized = {
        "role": "assistant",
        "content": choice_msg.content or "",
    }
    if choice_msg.tool_calls and allow_tools:
        serialized["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in choice_msg.tool_calls
        ]
    return serialized


def _call_ollama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Executes tool-calling completion via local Ollama."""
    import ollama

    print(f"       -> [OLLAMA DISPATCH] Calling local Ollama model: {settings.OLLAMA_MODEL}...")
    response = ollama.chat(
        model=settings.OLLAMA_MODEL,
        messages=messages,
        tools=tools_schema,
    )
    print(f"       <- [OLLAMA RESPONSE RECEIVED]")
    return _serialize_message(response["message"])


@trace_run(name="[Phase 2] LLM Reasoner", run_type="llm", tags=["phase-2", "llm"])
def llm_node(state: AgentState) -> Dict[str, Any]:
    """Calls configured LLM (Groq or Ollama) with tools and serializes messages cleanly."""
    messages = list(state["messages"])
    latest_role = messages[-1].get("role", "")
    retry_c = state.get("retry_count", 0)

    print("\n" + "-" * 80)
    if retry_c > 0:
        print(f"[STEP 2/6] >>> PHASE 2: LLM REASONER (Self-Correction Retry #{retry_c})")
        print(f"    -> Mode: Addressing AST Code Critic feedback to regenerate clean Python syntax")
    elif latest_role == "tool":
        print(f"[STEP 2/6] >>> PHASE 2: LLM REASONER (Context Synthesis Turn)")
        print(f"    -> Mode: Synthesizing final answer & code fix using retrieved context")
    else:
        print(f"[STEP 2/6] >>> PHASE 2: LLM REASONER (Initial Consultation)")
        print(f"    -> Mode: Analyzing query with Provider='{settings.MODEL_PROVIDER}', Model='{settings.GROQ_MODEL}'")
    print("-" * 80)

    step_count = state.get("step_count", 0)
    # Allow tool invocations for up to 2 rounds; after that, guide the LLM to synthesize the final answer
    allow_tools = (step_count < 2)
    if not allow_tools and latest_role == "tool":
        messages.append({
            "role": "system",
            "content": "You have retrieved relevant code context. Now synthesize and present your comprehensive final answer to the user."
        })

    try:
        if settings.MODEL_PROVIDER.lower() == "groq" or (settings.GROQ_API_KEY and settings.MODEL_PROVIDER != "ollama"):
            clean_message = _call_groq(messages, allow_tools=allow_tools)
        else:
            clean_message = _call_ollama(messages)
    except Exception as exc:
        print("\n" + "*" * 80)
        print(f"[ERROR IN PHASE 2: LLM NODE FAILURE] {type(exc).__name__}: {exc}")
        print("*" * 80 + "\n")
        clean_message = {
            "role": "assistant",
            "content": f"I encountered an error connecting to the model engine ({exc}). Please check your API key or network connection."
        }

    if clean_message.get("tool_calls"):
        tool_names = [tc['function']['name'] for tc in clean_message['tool_calls']]
        print(f"    -> LLM Decision: [INVOKE TOOLS] Selected tool(s): {tool_names}")
        print(f"    -> Next Destination: PHASE 3 (Tool Dispatcher)")
    else:
        print(f"    -> LLM Decision: [SYNTHESIZE ANSWER] Generated proposed answer / code fix")
        print(f"    -> Next Destination: PHASE 5 (AST Fix Critic)")

    new_messages = list(messages)
    new_messages.append(clean_message)

    return {"messages": new_messages}


@trace_run(name="[Phase 3] Tool Dispatcher", run_type="tool", tags=["phase-3", "tools"])
def tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes tool calls and accumulates investigated artifacts."""
    latest_message = state["messages"][-1]
    tool_calls = latest_message.get("tool_calls", [])

    print("\n" + "=" * 80)
    print(f"[STEP 3/6] +++ PHASE 3: TOOL DISPATCHER - Executing {len(tool_calls)} Tool Call(s)")
    print("=" * 80)

    new_tool_calls = list(state.get("tool_calls", []))
    new_tool_results = list(state.get("tool_results", []))
    new_messages = list(state["messages"])

    investigated_files = list(state.get("investigated_files", []))
    active_symbols = list(state.get("active_symbols", []))

    for idx, tool_call in enumerate(tool_calls, 1):
        call_id = tool_call.get("id") or f"call_{len(new_tool_calls)}"
        function = tool_call["function"]
        tool_name = function["name"]
        arguments = function.get("arguments", {})

        print(f"    [{idx}/{len(tool_calls)}] Dispatching to Tool: '{tool_name}' (Call ID: {call_id})")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"query": arguments}

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
            "tool_call_id": call_id,
            "name": tool_name,
            "content": raw_result_str
        })

    current_steps = state.get("step_count", 0) + 1
    print(f"    -> Investigated Files so far: {investigated_files}")
    print(f"    -> Active Code Symbols so far: {active_symbols}")
    print(f"    -> Next Destination: PHASE 4 (Context Cross-Checker)")

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
@trace_run(name="[Phase 4] Context Cross-Checker", run_type="chain", tags=["phase-4", "verification"])
def cross_check_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Cross-checks retrieved snippets before answering to prevent hallucinations.
    """
    question = state["question"]
    tool_results = state.get("tool_results", [])
    snippets = context_verifier.extract_search_documents(tool_results)

    print("\n" + "=" * 80)
    print(f"[STEP 4/6] ### PHASE 4: CONTEXT VERIFIER - Anti-Hallucination Guardrail")
    print("=" * 80)
    print(f"    -> Cross-checking {len(snippets)} retrieved snippet(s) against question...")

    is_verified, score, explanation = context_verifier.verify_relevance(question, snippets)

    status_str = "[PASS] High Alignment" if is_verified else "[WARN] Low Alignment"
    print(f"    -> Alignment Score: {score:.2f} / 1.00 | Status: {status_str}")
    if explanation:
        print(f"    -> Verification Notes: {explanation}")

    new_messages = list(state["messages"])
    if not is_verified:
        print(f"    [WARN IN PHASE 4] Injecting anti-hallucination constraint into system prompt.")
        guard_prompt = (
            f"[Context Cross-Check Warning: Relevance score was low ({score:.2f}). "
            f"Note: {explanation}. If the retrieved code does not clearly answer the user's question, "
            f"state clearly that the codebase does not provide this information rather than guessing.]"
        )
        new_messages.append({"role": "system", "content": guard_prompt})

    print(f"    -> Next Destination: PHASE 2 (LLM Reasoner for Context Synthesis)")
    return {
        "context_verified": is_verified,
        "context_confidence": score,
        "verification_notes": explanation,
        "messages": new_messages,
    }


@trace_run(name="[Phase 5] AST Fix Critic", run_type="chain", tags=["phase-5", "ast-validation"])
def validate_fix_node(state: AgentState) -> Dict[str, Any]:
    """Validates the proposed fix or code blocks generated by the LLM."""
    latest_message = state["messages"][-1]
    content = latest_message.get("content", "")

    print("\n" + "=" * 80)
    print(f"[STEP 5/6] ??? PHASE 5: AST FIX CRITIC - Code Syntax & Integrity Validator")
    print("=" * 80)
    print(f"    -> Inspecting generated Python markdown code blocks via 'ast.parse'...")

    is_valid, feedback = fix_validator.evaluate_proposed_fix(content)

    new_messages = list(state["messages"])
    current_retries = state.get("retry_count", 0)

    if is_valid:
        print(f"    -> AST Syntax Check: [PASS] All code blocks compile into valid, clean Python AST.")
        print(f"    -> Next Destination: PHASE 6 (Finalize & Persist)")
    else:
        current_retries += 1
        print("\n" + "*" * 80)
        print(f"[ERROR DETECTED IN PHASE 5: AST CODE CRITIC]")
        print(f"    -> Code Syntax Error: {feedback}")
        print(f"    -> Action: Triggering Self-Correction Retry Loop (Attempt #{current_retries} of {state.get('max_retries', 3)})")
        print(f"    -> Next Destination: PHASE 2 (LLM Reasoner to re-synthesize clean syntax)")
        print("*" * 80 + "\n")
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


@trace_run(name="[Phase 6: END] Finalize & Persist", run_type="chain", tags=["phase-6", "finalize"])
def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Extracts final answer and updates session debug notes."""
    print("\n" + "=" * 80)
    print(f"[STEP 6/6] <<< PHASE 6: END - Final Answer Assembly & Checkpoint Persistence")
    print("=" * 80)

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

    print(f"    -> Final Answer Synthesis: Complete ({len(final_answer)} characters)")
    print(f"    -> Context Cross-Check: Verified={state.get('context_verified', False)} (Confidence: {state.get('context_confidence', 0.0):.2f})")
    print(f"    -> Session State: Checkpointed to SQLite Database (data/sessions.db)")
    print(f"    -> Status: [WORKFLOW COMPLETE - SUCCESS]")
    print("=" * 80 + "\n")

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