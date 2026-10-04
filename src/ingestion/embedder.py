from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import CodeChunk


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

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

        print("Embedding model loaded.")

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