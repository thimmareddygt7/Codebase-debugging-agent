import chromadb


class CodeVectorStore:

    COLLECTION_NAME = "codebase"

    def __init__(
        self,
        persist_directory: str = "vector_store"
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )

    def reset(self):

        try:
            self.client.delete_collection(
                name=self.COLLECTION_NAME
            )
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )

    def add_chunks(
        self,
        ids: list[str],
        embeddings,
        documents: list[str],
        metadatas: list[dict]
    ):

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )

    def count(self) -> int:

        return self.collection.count()

    def get_chunk(
        self,
        chunk_id: str
    ):

        return self.collection.get(
            ids=[chunk_id],
            include=[
                "documents",
                "metadatas",
            ]
        )

    def search(
        self,
        query_embedding,
        top_k: int = 5
    ):

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ]
        )

        return results