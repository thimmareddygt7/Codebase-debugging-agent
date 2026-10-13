from fastapi import APIRouter, HTTPException
from app.api.schemas import DebugQueryRequest, DebugQueryResponse
from src.agent.graph import ask_codebase_agent

router = APIRouter()

@router.post("/ask", response_model=DebugQueryResponse)
def ask_agent(payload: DebugQueryRequest):
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