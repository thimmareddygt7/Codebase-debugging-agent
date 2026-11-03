"""FastAPI security dependencies: API-key auth and the command-execution kill switch."""
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.auth import key_is_valid, parse_api_keys
from src.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: Optional[str] = Security(_api_key_header)) -> None:
    """
    Fail-closed authentication.
    - Keys configured  -> request must send a matching X-API-Key header.
    - No keys configured -> 503, unless ALLOW_INSECURE_NO_AUTH=true (local dev only).
    """
    keys = parse_api_keys(settings.API_KEYS)
    if not keys:
        if settings.ALLOW_INSECURE_NO_AUTH:
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server authentication is not configured. Set AGENT_API_KEYS (min 16 chars).",
        )
    if not key_is_valid(provided, keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def require_command_execution_enabled() -> None:
    """Command execution (tests / git) is OFF unless explicitly enabled."""
    if not settings.ENABLE_COMMAND_EXECUTION:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Command execution is disabled on this deployment (ENABLE_COMMAND_EXECUTION=false).",
        )
