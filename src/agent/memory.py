import sqlite3
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.checkpoint.sqlite import SqliteSaver


class AgentState(TypedDict):
    question: str
    query_type: Optional[str]
    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_answer: str
    error: Optional[str]
    step_count: int
    retry_count: int
    max_retries: int
    fix_valid: bool
    validation_feedback: Optional[str]
    investigated_files: List[str]
    active_symbols: List[str]
    debug_notes: List[str]
    # v7a: Context cross-checking fields
    context_verified: bool
    context_confidence: float
    verification_notes: Optional[str]


class SessionManager:
    """
    Manages persistent SQLite-backed checkpoints for multi-turn debugging sessions.
    Guarantees session durability across process restarts.
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(self.conn)

    def get_checkpointer(self) -> SqliteSaver:
        """Returns the active SqliteSaver instance."""
        return self._checkpointer

    @staticmethod
    def get_thread_config(thread_id: str) -> dict:
        """Returns standard LangGraph configuration dictionary for a given thread_id."""
        return {"configurable": {"thread_id": thread_id}}