import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "groq")
    MODEL_NAME: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "vector_store")
    SESSION_DB_PATH: Path = PROJECT_ROOT / os.getenv("SESSION_DB_PATH", "data/sessions.db")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")

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