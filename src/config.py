import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "groq")
    MODEL_NAME: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "vector_store")
    SESSION_DB_PATH: Path = PROJECT_ROOT / os.getenv("SESSION_DB_PATH", "data/sessions.db")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")

    # --- API security (fail-closed by default) ---
    # Comma-separated API keys (each >= 16 chars). Clients send: X-API-Key: <key>
    API_KEYS: str = os.getenv("AGENT_API_KEYS", "")
    # Local development only: allow requests when no keys are configured.
    ALLOW_INSECURE_NO_AUTH: bool = os.getenv("ALLOW_INSECURE_NO_AUTH", "false").lower() in ("true", "1")
    # Test runner / git execution is OFF unless explicitly enabled.
    ENABLE_COMMAND_EXECUTION: bool = os.getenv("ENABLE_COMMAND_EXECUTION", "false").lower() in ("true", "1")
    # Comma-separated browser origins allowed by CORS.
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:8000")

    # --- Retrieval ---
    ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "true").lower() in ("true", "1")
    RERANK_OVERSAMPLE: int = int(os.getenv("RERANK_OVERSAMPLE", "3"))

    # --- Q&A agent limits (raise on paid tiers / larger context models) ---
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    TOOL_SNIPPET_CHARS: int = int(os.getenv("TOOL_SNIPPET_CHARS", "750"))
    TOOL_TOP_K_MAX: int = int(os.getenv("TOOL_TOP_K_MAX", "3"))

    # --- Repository access ---
    # Directory the agent debugs (empty = this project). Extra comma-separated roots
    # that /index and /debug/fix may read:
    TARGET_REPO_DIR: str = os.getenv("TARGET_REPO_DIR", "")
    INDEX_ALLOWED_ROOTS: str = os.getenv("INDEX_ALLOWED_ROOTS", "")

    # --- Auto-fix loop (reproduce -> patch -> re-test in a disposable workspace) ---
    SANDBOX_MODE: str = os.getenv("SANDBOX_MODE", "local")  # "local" or "docker"
    SANDBOX_DOCKER_IMAGE: str = os.getenv("SANDBOX_DOCKER_IMAGE", "python:3.11-slim")
    FIX_MAX_STEPS: int = int(os.getenv("FIX_MAX_STEPS", "12"))
    FIX_MAX_TOKENS: int = int(os.getenv("FIX_MAX_TOKENS", "2000"))
    FIX_TEST_TIMEOUT: int = int(os.getenv("FIX_TEST_TIMEOUT", "60"))
    FIX_MAX_SECONDS: int = int(os.getenv("FIX_MAX_SECONDS", "300"))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # LangSmith Observability
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "codebase-debugging-agent")
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "true").lower() in ("true", "1")

settings = Settings()

# Synchronize LangSmith / LangChain tracing environment variables
if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
