from pydantic import BaseModel, Field
from typing import List, Optional

class DebugQueryRequest(BaseModel):
    question: str = Field(..., description="Debugging question or problem statement.")
    thread_id: str = Field(default="default_session", description="Session ID for multi-turn persistence.")
    max_retries: int = Field(default=2, ge=1, le=5, description="Maximum AST validation fix attempts.")

class DebugQueryResponse(BaseModel):
    question: str
    final_answer: str
    investigated_files: List[str]
    active_symbols: List[str]
    context_verified: bool
    context_confidence: float
    fix_valid: bool
    error: Optional[str] = None