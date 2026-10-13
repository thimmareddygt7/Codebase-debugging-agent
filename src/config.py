import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "ollama")
    MODEL_NAME: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "vector_store")
    SESSION_DB_PATH: Path = PROJECT_ROOT / os.getenv("SESSION_DB_PATH", "data/sessions.db")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")

settings = Settings()