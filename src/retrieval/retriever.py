import logging

from src.config import settings
from src.ingestion.embedder import get_shared_embedder
from src.observability import trace_run
from src.retrieval.reranker import CodeReranker
from src.retrieval.vector_store import CodeVectorStore

logger = logging.getLogger(__name__)


class CodeRetriever:
    """
    Vector search with optional hybrid reranking.

    When reranking is on, the vector store is asked for ``top_k * oversample``
    candidates and ``CodeReranker`` (semantic score + lexical/symbol overlap)
    reorders them down to ``top_k``.
    """

    def __init__(
        self,
        persist_directory: str = "vector_store",
        embedder=None,
        vector_store=None,
        reranker=None,
        use_reranker: bool | None = None,
        oversample: int | None = None,
    ):
        self.embedder = embedder or get_shared_embedder()
        self.vector_store = vector_store or CodeVectorStore(persist_directory=persist_directory)
        self.use_reranker = settings.ENABLE_RERANKER if use_reranker is None else use_reranker
        self.reranker = reranker or CodeReranker()
        self.oversample = max(1, oversample or settings.RERANK_OVERSAMPLE)

    @trace_run(name="code_retriever_search", run_type="retriever")
    def search(self, query: str, top_k: int = 5, rerank: bool | None = None):
        do_rerank = self.use_reranker if rerank is None else rerank
        query_embedding = self.embedder.embed_text(query)
        candidates = top_k * self.oversample if do_rerank else top_k
        results = self.vector_store.search(query_embedding=query_embedding, top_k=candidates)
        if do_rerank:
            results = self.reranker.rerank(query, results, top_k=top_k)
        return results
