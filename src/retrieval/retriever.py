from src.ingestion.embedder import CodeEmbedder
from src.retrieval.vector_store import CodeVectorStore


class CodeRetriever:

    def __init__(
        self,
        persist_directory: str = "vector_store"
    ):

        print("Initializing code retriever...")

        self.embedder = CodeEmbedder()

        self.vector_store = CodeVectorStore(
            persist_directory=persist_directory
        )

        print("Code retriever initialized.")

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        # --------------------------------------------------
        # STEP 1: Convert user query into embedding
        # --------------------------------------------------

        query_embedding = self.embedder.embed_text(
            query
        )

        # --------------------------------------------------
        # STEP 2: Search vector store
        # --------------------------------------------------

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results