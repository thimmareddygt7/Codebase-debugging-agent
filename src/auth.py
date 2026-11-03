"""Pure (framework-free) API-key helpers, so they can be unit-tested without FastAPI."""
import hmac
import logging
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

MIN_KEY_LENGTH = 16


def parse_api_keys(raw: str, min_length: int = MIN_KEY_LENGTH) -> List[str]:
    """Parse a comma-separated key list, dropping empty and too-short (weak) keys."""
    keys: List[str] = []
    for candidate in (raw or "").split(","):
        key = candidate.strip()
        if not key:
            continue
        if len(key) < min_length:
            logger.warning("Ignoring an API key shorter than %d characters in AGENT_API_KEYS.", min_length)
            continue
        keys.append(key)
    return keys


def key_is_valid(provided: Optional[str], configured: Iterable[str]) -> bool:
    """Constant-time comparison against every configured key (no early exit)."""
    if not provided:
        return False
    valid = False
    for key in configured:
        if hmac.compare_digest(provided.encode("utf-8"), key.encode("utf-8")):
            valid = True
    return valid
