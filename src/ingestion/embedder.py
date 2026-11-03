import logging
import threading

from src.config import settings
from src.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


class CodeEmbeddingPreparer:

    def prepare_chunk(self, chunk: CodeChunk) -> str:

        parts = [
            f"File: {chunk.file_path}",
            f"Language: {chunk.language}",
            f"Type: {chunk.chunk_type}",
            f"Name: {chunk.name}",
            f"Qualified Name: {chunk.qualified_name}",
        ]

        if chunk.class_name:
            parts.append(f"Class: {chunk.class_name}")

        parts.append(
            f"Lines: {chunk.start_line}-{chunk.end_line}"
        )

        parts.append("")
        parts.append("Code:")
        parts.append(chunk.code)

        return "\n".join(parts)

    def prepare_chunks(
        self,
        chunks: list[CodeChunk]
    ) -> list[str]:

        return [
            self.prepare_chunk(chunk)
            for chunk in chunks
        ]


class CodeEmbedder:

    MODEL_NAME = settings.EMBEDDING_MODEL_NAME

    def __init__(self, model_name: str | None = None):
        # Imported lazily so this module (and CodeEmbeddingPreparer) can be used
        # and unit-tested without the heavy sentence-transformers dependency.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s ...", model_name or self.MODEL_NAME)
        self.model = SentenceTransformer(model_name or self.MODEL_NAME)
        logger.info("Embedding model loaded.")

    def embed_text(self, text: str):

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding

    def embed_texts(
        self,
        texts: list[str]
    ):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings


_shared_embedder = None
_shared_lock = threading.Lock()


def get_shared_embedder() -> "CodeEmbedder":
    """Process-wide embedder: the model is loaded once, not once per request/tool instance."""
    global _shared_embedder
    with _shared_lock:
        if _shared_embedder is None:
            _shared_embedder = CodeEmbedder()
        return _shared_embedder
