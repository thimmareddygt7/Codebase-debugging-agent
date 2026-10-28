import os
import re
from typing import List, Pattern

# Compiled regex patterns for detecting and redacting sensitive secrets
SECRET_PATTERNS: List[Pattern] = [
    # Groq API keys: gsk_...
    re.compile(r"gsk_[a-zA-Z0-9]{32,}", re.IGNORECASE),
    # LangSmith / LangChain keys: lsv2_pt_... or ls__...
    re.compile(r"lsv2_[a-zA-Z0-9_]{20,}", re.IGNORECASE),
    re.compile(r"ls__[a-zA-Z0-9]{20,}", re.IGNORECASE),
    # OpenAI API keys: sk-...
    re.compile(r"sk-[a-zA-Z0-9T3BlbkFJ]{20,}", re.IGNORECASE),
    # Anthropic API keys: sk-ant-...
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{30,}", re.IGNORECASE),
    # GitHub Personal Access Tokens: ghp_... or github_pat_...
    re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE),
    re.compile(r"github_pat_[a-zA-Z0-9_]{40,}", re.IGNORECASE),
    # Hugging Face tokens: hf_...
    re.compile(r"hf_[a-zA-Z0-9]{30,}", re.IGNORECASE),
    # Private Key blocks
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Generic bearer tokens: Bearer eyJ...
    re.compile(r"Bearer\s+eyJ[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    # Environment variable assignment with sensitive names: API_KEY=..., SECRET=..., PASSWORD=...
    re.compile(r"(?i)(api[_-]?key|secret|password|passwd|token|auth[_-]?key)\s*[:=]\s*['\"]?([^\s'\"`]+)['\"]?"),
]

REDACTION_MASK = "[REDACTED_SECRET]"


def redact_secrets(text: str) -> str:
    """
    Scans input text and replaces all matched secrets, tokens, and private credentials
    with a safe redaction mask. Safe for terminal printing, logs, and LangSmith traces.
    """
    if not text or not isinstance(text, str):
        return text

    sanitized = text

    # Redact known patterns
    for pattern in SECRET_PATTERNS:
        # Check if pattern has groups (like key=value assignments)
        if pattern.groups > 0:
            sanitized = pattern.sub(r"\1=" + REDACTION_MASK, sanitized)
        else:
            sanitized = pattern.sub(REDACTION_MASK, sanitized)

    # Redact active environment secrets dynamically if set in environment
    for key in ("GROQ_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        val = os.getenv(key, "")
        if val and len(val) > 6 and val in sanitized:
            sanitized = sanitized.replace(val, f"[{key}_REDACTED]")

    return sanitized


def contains_secrets(text: str) -> bool:
    """Returns True if the text contains any recognizable secret or unredacted API key."""
    if not text or not isinstance(text, str):
        return False

    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True

    for key in ("GROQ_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        val = os.getenv(key, "")
        if val and len(val) > 6 and val in text:
            return True

    return False
