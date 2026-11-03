import functools
from typing import Optional, Callable, Any, List, Dict
from src.config import settings

try:
    from langsmith import Client, traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    traceable = None


def is_tracing_enabled() -> bool:
    """Returns True if LangSmith API key is present and tracing is active."""
    return bool(LANGSMITH_AVAILABLE and settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING)


def get_langsmith_client() -> Optional[Any]:
    """Returns authenticated LangSmith Client instance or None."""
    if is_tracing_enabled():
        try:
            return Client(api_key=settings.LANGSMITH_API_KEY)
        except Exception:
            return None
    return None


def get_project_dashboard_url() -> str:
    """Returns direct web URL to the LangSmith project dashboard."""
    project = settings.LANGSMITH_PROJECT
    return f"https://smith.langchain.com/o/default/projects/p/{project}"


from src.guardrails.secrets import redact_secrets


def _sanitize_dict(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data or not isinstance(data, dict):
        return {}
    return {
        k: redact_secrets(str(v)) if isinstance(v, str) else v
        for k, v in data.items()
    }


def trace_run(
    name: Optional[str] = None,
    run_type: str = "chain",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Safe decorator for tracing functions in LangSmith with automatic secret redaction.
    Gracefully falls back to direct execution if LangSmith is disabled or unavailable.
    """
    def decorator(func: Callable) -> Callable:
        if is_tracing_enabled() and traceable:
            try:
                clean_tags = [redact_secrets(t) for t in (tags or [])]
                clean_meta = _sanitize_dict(metadata)
                return traceable(
                    name=name or func.__name__,
                    run_type=run_type,
                    tags=clean_tags,
                    metadata=clean_meta,
                )(func)
            except Exception:
                return func
        return func

    return decorator
