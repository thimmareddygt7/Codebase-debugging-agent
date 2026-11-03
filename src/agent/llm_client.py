"""
Minimal tool-calling LLM abstraction for the fix loop.

The loop depends only on the ``ToolCallingLLM`` protocol, so it can be driven by
Groq, Ollama, or a scripted fake in tests/benchmarks. Provider SDKs are imported
lazily so this module imports without them installed.
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from src.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """No usable LLM is configured (missing key / provider)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ToolCallingLLM(Protocol):
    def chat(self, messages: List[dict], tools: List[dict], max_tokens: int) -> LLMReply: ...


def assistant_message(reply: LLMReply) -> dict:
    """OpenAI-style assistant message (with tool_calls) to append to the transcript."""
    msg: Dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in reply.tool_calls
        ]
    return msg


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"_invalid_json": str(raw)[:200]}
    except (TypeError, ValueError):
        return {"_invalid_json": str(raw)[:200]}


class GroqToolLLM:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 fallback_model: Optional[str] = None, client: Any = None):
        self.model = model or settings.GROQ_MODEL
        self.fallback_model = fallback_model or settings.FALLBACK_MODEL
        if client is not None:
            self.client = client
        else:
            key = api_key or settings.GROQ_API_KEY
            if not key:
                raise LLMUnavailable("GROQ_API_KEY is not set.")
            from groq import Groq  # lazy import
            self.client = Groq(api_key=key)

    def chat(self, messages: List[dict], tools: List[dict], max_tokens: int) -> LLMReply:
        models = [self.model] + ([self.fallback_model] if self.fallback_model and self.fallback_model != self.model else [])
        last_exc: Optional[Exception] = None
        for model in models:
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, tools=tools, tool_choice="auto",
                    temperature=0.0, max_tokens=max_tokens,
                )
                break
            except Exception as exc:  # rate limit, request too large, model retired, 5xx ...
                last_exc = exc
                logger.warning("LLM call failed on %s (%s)%s", model, type(exc).__name__,
                               "; trying fallback model" if model != models[-1] else "")
        else:
            raise last_exc  # type: ignore[misc]

        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id or f"call_{i}", name=tc.function.name,
                     arguments=_parse_arguments(tc.function.arguments))
            for i, tc in enumerate(getattr(msg, "tool_calls", None) or [])
        ]
        usage = getattr(resp, "usage", None)
        return LLMReply(
            content=getattr(msg, "content", "") or "",
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class OllamaToolLLM:
    """Local models via Ollama. NOTE: tool-call quality varies a lot by model."""

    def __init__(self, model: Optional[str] = None, client: Any = None):
        self.model = model or settings.OLLAMA_MODEL
        if client is not None:
            self.client = client
        else:
            import ollama  # lazy import
            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    @staticmethod
    def _convert(messages: List[dict]) -> List[dict]:
        out = []
        for m in messages:
            if m.get("tool_calls"):
                m = dict(m)
                m["tool_calls"] = [
                    {"function": {"name": tc["function"]["name"],
                                  "arguments": _parse_arguments(tc["function"]["arguments"])}}
                    for tc in m["tool_calls"]
                ]
            out.append(m)
        return out

    def chat(self, messages: List[dict], tools: List[dict], max_tokens: int) -> LLMReply:
        resp = self.client.chat(
            model=self.model, messages=self._convert(messages), tools=tools,
            options={"num_predict": max_tokens, "temperature": 0.0},
        )
        get = (lambda o, k, d=None: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d))
        message = get(resp, "message", {})
        calls = []
        for i, tc in enumerate(get(message, "tool_calls", None) or []):
            fn = get(tc, "function", {})
            calls.append(ToolCall(id=f"call_{i}", name=get(fn, "name", ""),
                                  arguments=_parse_arguments(get(fn, "arguments", {}))))
        return LLMReply(
            content=get(message, "content", "") or "", tool_calls=calls,
            prompt_tokens=get(resp, "prompt_eval_count", 0) or 0,
            completion_tokens=get(resp, "eval_count", 0) or 0,
        )


def get_default_llm() -> ToolCallingLLM:
    if settings.MODEL_PROVIDER == "ollama":
        return OllamaToolLLM()
    return GroqToolLLM()
