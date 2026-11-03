"""
Code snippet reranker using reciprocal rank fusion, lexical alignment, and symbol matching.
"""
from typing import Dict, Any, List, Optional
import math


class CodeReranker:
    """
    Reranks vector search results by combining semantic vector distance
    with lexical token overlap and exact symbol match boosts.
    """

    def __init__(self, semantic_weight: float = 0.6, lexical_weight: float = 0.4):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def _extract_query_terms(self, query: str) -> List[str]:
        """Extracts meaningful terms from the query for keyword scoring."""
        stopwords = {
            "where", "what", "which", "how", "why", "the", "and", "does", "for",
            "with", "from", "are", "can", "you", "show", "find", "code", "function",
        }
        tokens = query.lower().replace("()", " ").replace(".", " ").replace("_", " ").split()
        return [t.strip("?,.:;()[]'\"") for t in tokens if len(t) > 2 and t not in stopwords]

    def _calculate_lexical_score(self, query_terms: List[str], doc_text: str, metadata: Dict[str, Any]) -> float:
        """Calculates a normalized lexical overlap score between query terms and snippet."""
        if not query_terms:
            return 0.5

        doc_lower = (doc_text or "").lower()
        name = str(metadata.get("name", "")).lower()
        qualified_name = str(metadata.get("qualified_name", "")).lower()
        file_path = str(metadata.get("file_path", "")).lower()

        # Term frequency score
        matched_terms = 0
        for term in query_terms:
            if term in doc_lower or term in name or term in file_path:
                matched_terms += 1

        base_ratio = matched_terms / len(query_terms)

        # Boost if the function or class name exactly contains one of the query terms
        symbol_boost = 0.0
        for term in query_terms:
            if term == name or term in qualified_name:
                symbol_boost = 0.3
                break

        return min(1.0, base_ratio + symbol_boost)

    def rerank(
        self,
        query: str,
        results: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reranks ChromaDB query results dict and returns top_k reranked results.
        Expected results structure: { 'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]] }
        """
        if not results or not results.get("ids") or not results["ids"][0]:
            return results

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        query_terms = self._extract_query_terms(query)
        scored_items = []

        for idx, item_id in enumerate(ids):
            doc = documents[idx] if idx < len(documents) else ""
            meta = metadatas[idx] if idx < len(metadatas) else {}
            dist = distances[idx] if idx < len(distances) else 1.0

            # Convert cosine/L2 distance to 0..1 similarity score
            semantic_score = 1.0 / (1.0 + max(0.0, float(dist)))

            # Lexical score
            lexical_score = self._calculate_lexical_score(query_terms, doc, meta)

            # Combined hybrid reranking score
            hybrid_score = (self.semantic_weight * semantic_score) + (self.lexical_weight * lexical_score)

            scored_items.append({
                "id": item_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "rerank_score": hybrid_score,
            })

        # Sort descending by hybrid rerank score
        scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            scored_items = scored_items[:top_k]

        return {
            "ids": [[item["id"] for item in scored_items]],
            "documents": [[item["document"] for item in scored_items]],
            "metadatas": [[item["metadata"] for item in scored_items]],
            "distances": [[item["distance"] for item in scored_items]],
            "rerank_scores": [[item["rerank_score"] for item in scored_items]],
        }
