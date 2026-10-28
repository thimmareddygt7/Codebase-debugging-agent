import os
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.ingestion.loader import CodebaseLoader
from src.ingestion.chunker import CodebaseChunker
from src.ingestion.embedder import CodeEmbeddingPreparer, CodeEmbedder
from src.retrieval.vector_store import CodeVectorStore
from src.config import settings


class CodebaseIndexer:
    """
    Manages indexing and re-indexing of code repositories into ChromaDB
    with status reporting and thread-safe background execution.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, persist_directory: str = "vector_store"):
        if getattr(self, "_initialized", False):
            return

        self.persist_directory = persist_directory
        self.state: Dict[str, Any] = {
            "is_indexing": False,
            "status": "idle",  # idle, in_progress, completed, failed
            "target_path": None,
            "files_count": 0,
            "chunks_count": 0,
            "records_stored": 0,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "last_indexed_at": None
        }
        self._thread: Optional[threading.Thread] = None
        self._embedder: Optional[CodeEmbedder] = None
        self._vector_store: Optional[CodeVectorStore] = None
        self._initialized = True

    def get_status(self) -> Dict[str, Any]:
        """Returns the current indexing status and last run metadata."""
        current = dict(self.state)
        # If vector store exists, get current live count
        if not current["is_indexing"] and current["records_stored"] == 0:
            try:
                vs = CodeVectorStore(persist_directory=self.persist_directory)
                current["records_stored"] = vs.count()
            except Exception:
                pass
        return current

    def index_repository(
        self,
        target_path: str = ".",
        reset: bool = True,
        async_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Triggers repository indexing either synchronously or in a background worker thread.
        """
        with self._lock:
            if self.state["is_indexing"]:
                return {
                    "success": False,
                    "message": "Indexing is already in progress.",
                    "status": self.get_status()
                }

            self.state["is_indexing"] = True
            self.state["status"] = "in_progress"
            self.state["target_path"] = target_path
            self.state["error"] = None
            self.state["started_at"] = datetime.now(timezone.utc).isoformat()
            self.state["completed_at"] = None

        if async_mode:
            self._thread = threading.Thread(
                target=self._run_indexing,
                args=(target_path, reset),
                daemon=True
            )
            self._thread.start()
            return {
                "success": True,
                "message": f"Background repository indexing started for '{target_path}'.",
                "status": self.get_status()
            }
        else:
            self._run_indexing(target_path, reset)
            return {
                "success": self.state["status"] == "completed",
                "message": f"Repository indexing finished with status '{self.state['status']}'.",
                "status": self.get_status()
            }

    def _run_indexing(self, target_path: str, reset: bool):
        """Worker executing parsing, chunking, embedding, and storage."""
        try:
            resolved_root = str(Path(target_path).resolve())

            # 1. Load files
            loader = CodebaseLoader(resolved_root)
            files = loader.get_source_files()

            # 2. Chunk files
            chunker = CodebaseChunker()
            chunks = chunker.chunk_codebase(files)

            if not chunks:
                with self._lock:
                    self.state["is_indexing"] = False
                    self.state["status"] = "completed"
                    self.state["files_count"] = len(files)
                    self.state["chunks_count"] = 0
                    self.state["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.state["last_indexed_at"] = self.state["completed_at"]
                return

            # 3. Prepare chunk text representation
            preparer = CodeEmbeddingPreparer()
            prepared_texts = preparer.prepare_chunks(chunks)

            # 4. Generate embeddings
            if self._embedder is None:
                self._embedder = CodeEmbedder()
            embeddings = self._embedder.embed_texts(prepared_texts)

            # 5. Prepare metadatas & IDs
            ids = [c.chunk_id for c in chunks]
            metadatas = [
                {
                    "file_path": c.file_path,
                    "language": c.language,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "qualified_name": c.qualified_name,
                    "class_name": c.class_name or "",
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in chunks
            ]

            # 6. Store in ChromaDB
            vs = CodeVectorStore(persist_directory=self.persist_directory)
            if reset:
                vs.reset()
            vs.add_chunks(
                ids=ids,
                embeddings=embeddings,
                documents=prepared_texts,
                metadatas=metadatas
            )

            with self._lock:
                self.state["is_indexing"] = False
                self.state["status"] = "completed"
                self.state["files_count"] = len(files)
                self.state["chunks_count"] = len(chunks)
                self.state["records_stored"] = vs.count()
                self.state["completed_at"] = datetime.now(timezone.utc).isoformat()
                self.state["last_indexed_at"] = self.state["completed_at"]

        except Exception as exc:
            with self._lock:
                self.state["is_indexing"] = False
                self.state["status"] = "failed"
                self.state["error"] = str(exc)
                self.state["completed_at"] = datetime.now(timezone.utc).isoformat()


# Global indexer instance
indexer_instance = CodebaseIndexer()
