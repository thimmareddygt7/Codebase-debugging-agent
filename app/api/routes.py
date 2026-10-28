import os
import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.api.schemas import (
    HealthResponse,
    ReadyResponse,
    DebugQueryRequest,
    DebugQueryResponse,
    SearchRequest,
    SearchResponse,
    DebugInvestigateRequest,
    DebugInvestigateResponse,
    IndexRequest,
    IndexResponse,
    IndexStatusResponse,
    AgentRunResponse,
    AgentRunEventsResponse,
    FixProposeRequest,
    FixResponse,
    FixDiffResponse,
    FixApproveRequest,
    FixRejectRequest,
    FixApplyRequest,
    FixApplyResponse,
    FixRollbackResponse,
    TestRunRequest,
    TestRunResponse,
    AuditLogsResponse,
    EvaluationResponse,
)

from src.config import settings
from src.agent.graph import ask_codebase_agent, agent_graph, session_manager
from src.agent.tools import CodeSearchTool, DocSearchTool
from src.retrieval.vector_store import CodeVectorStore
from src.ingestion.indexer import indexer_instance
from src.guardrails import (
    fix_manager,
    CommandSafetyGuardrail,
    AuditLogger,
    FileSystemGuardrail,
    redact_secrets,
)
from src.evaluation import RagasEvaluator

router = APIRouter()
cmd_guard = CommandSafetyGuardrail()
audit_logger = AuditLogger()
fs_guard = FileSystemGuardrail()

# In-memory test runs cache
_TEST_RUNS: Dict[str, Dict[str, Any]] = {}


# ===========================================================================
# STAGE 1: Core API Endpoints
# ===========================================================================

@router.get("/health", response_model=HealthResponse, tags=["Core API"])
def get_health():
    """Check whether the API is running."""
    return HealthResponse(
        status="ok",
        service="codebase-debugging-agent",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/ready", response_model=ReadyResponse, tags=["Core API"])
def get_ready():
    """Check whether required components (vector store, LLM, retriever) are ready."""
    components = {
        "vector_store": False,
        "llm": False,
        "retriever": False,
    }
    details = {}

    # 1. Vector Store check
    try:
        vs = CodeVectorStore(persist_directory="vector_store")
        records_count = vs.count()
        components["vector_store"] = True
        details["vector_store"] = {
            "ready": True,
            "records_count": records_count,
            "collection": vs.COLLECTION_NAME
        }
    except Exception as exc:
        details["vector_store"] = {"ready": False, "error": str(exc)}

    # 2. LLM Configuration check
    if settings.MODEL_PROVIDER == "groq":
        has_key = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
        components["llm"] = has_key
        details["llm"] = {
            "provider": "groq",
            "model": settings.GROQ_MODEL,
            "has_api_key": has_key
        }
    else:
        components["llm"] = True
        details["llm"] = {
            "provider": "ollama",
            "model": settings.OLLAMA_MODEL,
            "endpoint": settings.OLLAMA_BASE_URL
        }

    # 3. Retriever readiness
    components["retriever"] = components["vector_store"] and components["llm"]
    details["retriever"] = {"ready": components["retriever"]}

    all_ready = all(components.values())
    return ReadyResponse(
        status="ready" if all_ready else "degraded",
        components=components,
        details=details
    )


@router.post("/ask", response_model=DebugQueryResponse, tags=["Core API"])
def ask_agent(payload: DebugQueryRequest):
    """Ask a question about the codebase using RAG and LangGraph state machine."""
    try:
        result = ask_codebase_agent(
            question=payload.question,
            thread_id=payload.thread_id,
            max_retries=payload.max_retries
        )
        return DebugQueryResponse(
            question=result.get("question", payload.question),
            final_answer=result.get("final_answer", ""),
            investigated_files=result.get("investigated_files", []),
            active_symbols=result.get("active_symbols", []),
            context_verified=result.get("context_verified", False),
            context_confidence=result.get("context_confidence", 0.0),
            fix_valid=result.get("fix_valid", True),
            error=result.get("error")
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/search", response_model=SearchResponse, tags=["Core API"])
def search_codebase(payload: SearchRequest):
    """Search for relevant code chunks, functions, classes, and documentation."""
    results = []

    if payload.search_type in ("code", "all"):
        try:
            code_tool = CodeSearchTool()
            code_results = code_tool.search(query=payload.query, top_k=payload.top_k)
            for r in code_results:
                results.append({
                    "source": "code",
                    "file_path": r.get("file_path"),
                    "name": r.get("chunk_name"),
                    "qualified_name": r.get("qualified_name"),
                    "lines": f"{r.get('start_line')}-{r.get('end_line')}",
                    "distance": r.get("distance"),
                    "content": r.get("code")
                })
        except Exception as exc:
            pass

    if payload.search_type in ("doc", "all") and len(results) < payload.top_k:
        try:
            doc_tool = DocSearchTool()
            doc_results = doc_tool.search(query=payload.query)
            for r in doc_results:
                results.append({
                    "source": "doc",
                    "file_path": r.get("file_path"),
                    "name": Path(r.get("file_path", "")).name,
                    "lines": f"line {r.get('line')}",
                    "distance": None,
                    "content": r.get("content")
                })
        except Exception:
            pass

    return SearchResponse(
        query=payload.query,
        total_results=len(results),
        results=results[:payload.top_k]
    )


@router.post("/debug", response_model=DebugInvestigateResponse, tags=["Core API"])
def debug_investigate(payload: DebugInvestigateRequest):
    """Investigate a bug and return a diagnosis with supporting code references."""
    thread_id = payload.thread_id or f"debug_{uuid.uuid4().hex[:8]}"

    # Formulate structured bug diagnosis prompt
    prompt_parts = [
        f"BUG INVESTIGATION REQUEST:\nProblem Description:\n{payload.bug_description}"
    ]
    if payload.error_trace:
        prompt_parts.append(f"\nError Stack Trace:\n{payload.error_trace}")
    if payload.suspected_files:
        prompt_parts.append(f"\nSuspected Files: {', '.join(payload.suspected_files)}")
    prompt_parts.append(
        "\nPlease investigate the root cause, search the codebase for the relevant implementations, "
        "and provide a detailed diagnosis with exact line references and supporting code snippets."
    )

    full_query = "\n".join(prompt_parts)

    try:
        agent_out = ask_codebase_agent(question=full_query, thread_id=thread_id)

        # Extract supporting code references from tool results
        supporting = []
        for tr in agent_out.get("tool_results", []):
            try:
                res_obj = json.loads(tr.get("result", "{}"))
                for chunk in res_obj.get("results", []):
                    supporting.append({
                        "file_path": chunk.get("file_path"),
                        "name": chunk.get("chunk_name") or chunk.get("name"),
                        "code": chunk.get("code") or chunk.get("content")
                    })
            except Exception:
                pass

        return DebugInvestigateResponse(
            bug_description=payload.bug_description,
            diagnosis=agent_out.get("final_answer", ""),
            supporting_references=supporting[:5],
            investigated_files=agent_out.get("investigated_files", []),
            active_symbols=agent_out.get("active_symbols", []),
            context_verified=agent_out.get("context_verified", False),
            error=agent_out.get("error")
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# STAGE 2: Repository Indexing Endpoints
# ===========================================================================

@router.post("/index", response_model=IndexResponse, tags=["Repository Indexing"])
def trigger_indexing(payload: IndexRequest):
    """Index or re-index a target repository."""
    result = indexer_instance.index_repository(
        target_path=payload.target_path,
        reset=payload.reset_index,
        async_mode=payload.async_mode
    )
    return IndexResponse(
        success=result["success"],
        message=result["message"],
        status=result["status"]
    )


@router.get("/index/status", response_model=IndexStatusResponse, tags=["Repository Indexing"])
def get_indexing_status():
    """Check indexing progress and the last indexing result."""
    current = indexer_instance.get_status()
    return IndexStatusResponse(
        is_indexing=current.get("is_indexing", False),
        status=current.get("status", "idle"),
        target_path=current.get("target_path"),
        files_count=current.get("files_count", 0),
        chunks_count=current.get("chunks_count", 0),
        records_stored=current.get("records_stored", 0),
        started_at=current.get("started_at"),
        completed_at=current.get("completed_at"),
        last_indexed_at=current.get("last_indexed_at"),
        error=current.get("error")
    )


# ===========================================================================
# STAGE 3: LangGraph Workflow & Fix Proposal Endpoints
# ===========================================================================

@router.get("/agent/runs/{run_id}", response_model=AgentRunResponse, tags=["Workflow & Fixes"])
def get_agent_run(run_id: str):
    """Track a debugging workflow run from start to finish."""
    config = session_manager.get_thread_config(run_id)
    snapshot = agent_graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent workflow run '{run_id}' not found."
        )

    vals = snapshot.values
    return AgentRunResponse(
        run_id=run_id,
        status="completed" if vals.get("final_answer") else "in_progress",
        question=vals.get("question", ""),
        final_answer=vals.get("final_answer", ""),
        step_count=vals.get("step_count", 0),
        retry_count=vals.get("retry_count", 0),
        investigated_files=vals.get("investigated_files", []),
        active_symbols=vals.get("active_symbols", []),
        fix_valid=vals.get("fix_valid", True),
        context_verified=vals.get("context_verified", False),
        error=vals.get("error")
    )


@router.get("/agent/runs/{run_id}/events", response_model=AgentRunEventsResponse, tags=["Workflow & Fixes"])
def get_agent_run_events(run_id: str):
    """Retrieve workflow events, tool calls, and approval status."""
    config = session_manager.get_thread_config(run_id)
    events = []

    try:
        history = list(agent_graph.get_state_history(config))
        for idx, snapshot in enumerate(history):
            node_name = snapshot.metadata.get("source", "unknown") if snapshot.metadata else "unknown"
            vals = snapshot.values
            events.append({
                "step_index": idx + 1,
                "node": node_name,
                "tool_calls": vals.get("tool_calls", []),
                "tool_results_count": len(vals.get("tool_results", [])),
                "context_verified": vals.get("context_verified", False),
                "fix_valid": vals.get("fix_valid", True),
                "timestamp": snapshot.created_at if hasattr(snapshot, "created_at") else None
            })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read workflow history: {exc}")

    return AgentRunEventsResponse(
        run_id=run_id,
        events_count=len(events),
        events=events
    )


@router.post("/fix/propose", response_model=FixResponse, tags=["Workflow & Fixes"])
def propose_code_fix(payload: FixProposeRequest):
    """
    Generate a proposed fix WITHOUT changing files.
    Validates patch boundaries, diff size, and computes target file hashes.
    """
    target_path = Path(payload.target_file)
    is_safe, resolved_path, msg = fs_guard.sanitize_and_resolve(target_path)
    if not is_safe:
        raise HTTPException(status_code=400, detail=msg)

    new_content = payload.new_content

    # If new_content was not supplied manually, synthesize via LLM
    if not new_content:
        original = resolved_path.read_text(encoding="utf-8") if resolved_path.exists() else ""
        prompt = (
            f"You are a code refactoring engineer. Provide the complete updated Python code for {resolved_path.name} "
            f"to solve this bug:\n{payload.problem_description}\n\n"
            f"Original Content:\n{original}\n\n"
            "Return the full replacement file content inside a single ```python ... ``` markdown block only."
        )
        from src.rag.llm import get_llm
        raw_code = get_llm().generate(prompt)
        import re
        match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", raw_code)
        new_content = match.group(1) if match else raw_code

    result = fix_manager.propose_fix(
        target_file=payload.target_file,
        new_content=new_content,
        problem_description=payload.problem_description,
        rationale=payload.rationale or "Automated fix proposed via API",
        run_id=payload.run_id
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))

    return FixResponse(
        success=True,
        message=result["message"],
        fix=result["fix"]
    )


@router.get("/fix/{fix_id}", response_model=FixResponse, tags=["Workflow & Fixes"])
def get_fix(fix_id: str):
    """Retrieve a proposed fix and its current lifecycle status."""
    fix = fix_manager.get_fix(fix_id)
    if not fix:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fix '{fix_id}' not found."
        )
    return FixResponse(
        success=True,
        message=f"Fix '{fix_id}' retrieved.",
        fix=fix
    )


@router.get("/fix/{fix_id}/diff", response_model=FixDiffResponse, tags=["Workflow & Fixes"])
def get_fix_diff(fix_id: str):
    """Review the exact proposed changes in unified diff format."""
    diff_data = fix_manager.get_diff(fix_id)
    if not diff_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fix '{fix_id}' not found."
        )
    return FixDiffResponse(**diff_data)


# ===========================================================================
# STAGE 4: Approval and Safe Changes Endpoints
# ===========================================================================

@router.post("/fix/{fix_id}/approve", response_model=FixResponse, tags=["Approval & Safe Changes"])
def approve_fix(fix_id: str, payload: FixApproveRequest):
    """Explicitly approve the exact proposed patch."""
    result = fix_manager.approve_fix(
        fix_id=fix_id,
        approved_by=payload.approved_by,
        notes=payload.notes
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result["message"])
    return FixResponse(
        success=True,
        message=result["message"],
        fix=result.get("fix")
    )


@router.post("/fix/{fix_id}/reject", response_model=FixResponse, tags=["Approval & Safe Changes"])
def reject_fix(fix_id: str, payload: FixRejectRequest):
    """Reject a proposed patch with a reason."""
    result = fix_manager.reject_fix(
        fix_id=fix_id,
        reason=payload.reason
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result["message"])
    return FixResponse(
        success=True,
        message=result["message"],
        fix=result.get("fix")
    )


@router.post("/fix/{fix_id}/apply", response_model=FixApplyResponse, tags=["Approval & Safe Changes"])
def apply_fix(fix_id: str, payload: FixApplyRequest):
    """
    Apply an approved patch after safety checks and staleness verification.
    ENFORCES: Explicit human approval is required. If unapproved or stale, returns error.
    """
    result = fix_manager.apply_fix(
        fix_id=fix_id,
        run_tests_command=payload.run_tests_command
    )

    if not result.get("success"):
        status_code = (
            status.HTTP_403_FORBIDDEN
            if result.get("status") in ("APPROVAL_REQUIRED", "STALE_PATCH")
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=result["message"])

    return FixApplyResponse(
        success=True,
        message=result["message"],
        fix=result.get("fix"),
        test_output=result.get("test_output")
    )


@router.post("/fix/{fix_id}/rollback", response_model=FixRollbackResponse, tags=["Approval & Safe Changes"])
def rollback_fix(fix_id: str):
    """Attempt to revert an applied patch safely using its backup copy."""
    result = fix_manager.rollback_fix(fix_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result["message"])
    return FixRollbackResponse(
        success=True,
        message=result["message"],
        fix=result.get("fix")
    )


# ===========================================================================
# STAGE 5: Testing, Workflow, and Monitoring Endpoints
# ===========================================================================

@router.post("/tests/run", response_model=TestRunResponse, tags=["Testing & Monitoring"])
def run_approved_test(payload: TestRunRequest):
    """
    Run an approved test command against the target repository.
    Enforces strict command allowlist, execution timeouts, and resource caps.
    """
    is_allowed, requires_confirm, rationale = cmd_guard.is_command_allowed(payload.test_command)
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security Guardrail: {rationale}"
        )

    test_run_id = f"test_{uuid.uuid4().hex[:8]}"
    start_t = time.time()

    success, code, stdout, stderr = cmd_guard.execute_safe_command(
        command_str=payload.test_command,
        timeout=payload.timeout_seconds
    )
    duration = round(time.time() - start_t, 2)

    run_record = {
        "test_run_id": test_run_id,
        "command": payload.test_command,
        "success": success,
        "returncode": code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_seconds": duration,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    _TEST_RUNS[test_run_id] = run_record

    # Log to audit trail
    audit_logger.log_event(
        event_type="TEST_EXECUTION",
        tests_executed=payload.test_command,
        test_outcome="PASSED" if success else "FAILED",
        extra_details={"test_run_id": test_run_id, "duration": duration, "returncode": code}
    )

    return TestRunResponse(
        test_run_id=test_run_id,
        command=payload.test_command,
        success=success,
        returncode=code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration
    )


@router.get("/tests/{run_id}", response_model=TestRunResponse, tags=["Testing & Monitoring"])
def get_test_run(run_id: str):
    """Retrieve test results and execution status for a specific test run ID."""
    if run_id in _TEST_RUNS:
        return TestRunResponse(**_TEST_RUNS[run_id])

    # Search audit log fallback
    audit_file = settings.PROJECT_ROOT / "data" / "audit_log.jsonl"
    if audit_file.exists():
        for line in audit_file.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("details", {}).get("test_run_id") == run_id:
                    return TestRunResponse(
                        test_run_id=run_id,
                        command=rec.get("tests_executed", ""),
                        success=(rec.get("test_outcome") == "PASSED"),
                        returncode=rec.get("details", {}).get("returncode", 0),
                        stdout="[Archived run outcome: " + rec.get("test_outcome", "") + "]",
                        stderr="",
                        duration_seconds=rec.get("details", {}).get("duration", 0.0)
                    )
            except Exception:
                pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Test run '{run_id}' not found."
    )


@router.get("/audit/logs", response_model=AuditLogsResponse, tags=["Testing & Monitoring"])
def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None)
):
    """Review actions taken by the agent, approvals, file changes, and test runs."""
    audit_file = settings.PROJECT_ROOT / "data" / "audit_log.jsonl"
    if not audit_file.exists():
        return AuditLogsResponse(total_logs=0, count=0, logs=[])

    all_logs = []
    for line in audit_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                record = json.loads(line)
                if event_type and record.get("event_type") != event_type:
                    continue
                all_logs.append(record)
            except Exception:
                pass

    # Reverse chronological order
    all_logs.reverse()
    sliced = all_logs[offset : offset + limit]

    return AuditLogsResponse(
        total_logs=len(all_logs),
        count=len(sliced),
        logs=sliced
    )


@router.get("/evaluation", response_model=EvaluationResponse, tags=["Testing & Monitoring"])
def get_or_run_evaluation(
    run_now: bool = Query(default=False, description="Run live evaluation on sample query instead of reading cache.")
):
    """Run or retrieve agent evaluation results across Context, GT, and Retrieval metrics."""
    report_file = settings.PROJECT_ROOT / "data" / "ragas_report.json"

    if run_now:
        # Run live evaluation on first benchmark item
        dataset_path = settings.PROJECT_ROOT / "data" / "eval_dataset.json"
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Benchmark dataset not found.")

        samples = json.loads(dataset_path.read_text(encoding="utf-8"))[:1]
        sample = samples[0]

        evaluator = RagasEvaluator()
        agent_out = ask_codebase_agent(question=sample["question"], thread_id="eval_run_now")
        answer = agent_out.get("final_answer", "")

        deep_report = evaluator.evaluate_deep(
            question=sample["question"],
            answer=answer,
            contexts=[sample["ground_truth"]],
            ground_truth=sample["ground_truth"]
        )

        return EvaluationResponse(
            report_source="live_evaluation_run",
            summary={
                "composite_score": deep_report.composite_score,
                "summary": deep_report.summary,
                "faithfulness": deep_report.context_dependent.faithfulness,
                "context_utilization": deep_report.context_dependent.context_utilization,
                "answer_correctness": deep_report.ground_truth_dependent.llm_judge.answer_correctness if deep_report.ground_truth_dependent else None,
                "context_recall": deep_report.retrieval_dependent.context_recall,
                "context_precision": deep_report.retrieval_dependent.context_precision
            },
            detailed_samples=[{
                "question": sample["question"],
                "report": deep_report.to_dict()
            }]
        )

    # Return cached report
    if report_file.exists():
        try:
            cached = json.loads(report_file.read_text(encoding="utf-8"))
            return EvaluationResponse(
                report_source="cached_report_file",
                summary=cached.get("summary", {}),
                detailed_samples=cached.get("detailed_results", [])
            )
        except Exception:
            pass

    return EvaluationResponse(
        report_source="no_prior_evaluation",
        summary={"message": "No evaluation has been run yet. Call with ?run_now=true to execute."},
        detailed_samples=[]
    )