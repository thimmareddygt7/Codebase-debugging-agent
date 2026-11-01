"""
Corrective RAG (CRAG) pipeline for codebase retrieval and context verification.
Cross-checks retrieved code context and triggers corrective actions for low-relevance snippets.
"""
from typing import Dict, Any, List, Tuple
from src.agent.context_verifier import ContextCrossChecker


class CorrectiveRAG:
    """
    Implements Corrective RAG (CRAG) evaluation on retrieved codebase chunks.
    Assesses confidence, filters irrelevant context, and suggests refined search terms.
    """

    def __init__(self, confidence_threshold: float = 0.30):
        self.confidence_threshold = confidence_threshold
        self.checker = ContextCrossChecker()

    def evaluate_and_correct(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates retrieved chunks for relevance to query.
        Returns:
            {
                "is_sufficient": bool,
                "confidence": float,
                "action": "CORRECT" | "PROCEED" | "REWRITE_QUERY",
                "explanation": str,
                "filtered_documents": List[Dict[str, Any]],
                "refined_query_terms": List[str]
            }
        """
        is_sufficient, confidence, explanation = self.checker.verify_relevance(
            question=query,
            retrieved_items=retrieved_documents
        )

        stopwords = {"where", "what", "which", "how", "why", "the", "and", "does", "for", "with", "from", "are"}
        query_terms = [
            token.lower().strip("?,.:;()[]'\"")
            for token in query.split()
            if len(token) > 2 and token.lower() not in stopwords
        ]

        if not retrieved_documents:
            return {
                "is_sufficient": False,
                "confidence": 0.0,
                "action": "REWRITE_QUERY",
                "explanation": "No documents retrieved for query.",
                "filtered_documents": [],
                "refined_query_terms": query_terms
            }

        if confidence >= self.confidence_threshold:
            action = "PROCEED"
        else:
            action = "CORRECT"

        return {
            "is_sufficient": is_sufficient,
            "confidence": confidence,
            "action": action,
            "explanation": explanation,
            "filtered_documents": retrieved_documents if is_sufficient else [],
            "refined_query_terms": query_terms
        }
