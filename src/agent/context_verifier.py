import json
from typing import Dict, Any, List, Tuple

class ContextCrossChecker:
    """
    Evaluates whether retrieved tools/RAG context contains adequate evidence 
    to answer the user's codebase query before final synthesis.
    """

    @staticmethod
    def extract_search_documents(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses all successful search results from tool execution history."""
        extracted = []
        for tr in tool_results:
            raw = tr.get("result", "")
            try:
                parsed = json.loads(raw)
                if parsed.get("success"):
                    for item in parsed.get("results", []):
                        extracted.append(item)
            except Exception:
                continue
        return extracted

    def verify_relevance(self, question: str, retrieved_items: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        Cross-checks retrieved snippets against key query tokens.
        Returns: (is_sufficient, confidence_score, explanation)
        """
        if not retrieved_items:
            return False, 0.0, "No context items retrieved from tools."

        # Extract search keywords (tokens > 2 chars, ignoring common filler words)
        stopwords = {"where", "what", "which", "how", "why", "the", "and", "does", "for", "with", "from", "are"}
        query_terms = [
            token.lower().strip("?,.:;()[]'\"")
            for token in question.split()
            if len(token) > 2 and token.lower() not in stopwords
        ]

        if not query_terms:
            return True, 1.0, "Query terms non-restrictive; proceeding with context."

        matched_counts = []
        for item in retrieved_items:
            # Combine all text fields present in the snippet
            content_text = (
                item.get("code", "") + " " +
                item.get("qualified_name", "") + " " +
                item.get("chunk_name", "") + " " +
                item.get("file_path", "") + " " +
                item.get("content", "")
            ).lower()

            matches = sum(1 for term in query_terms if term in content_text)
            matched_counts.append(matches / len(query_terms))

        best_score = max(matched_counts) if matched_counts else 0.0

        # Cross-check confidence threshold (at least 30% of query keywords present in best snippet)
        if best_score >= 0.30:
            return True, round(best_score, 2), f"Retrieved context verified (relevance score: {best_score:.2f})."
        else:
            return False, round(best_score, 2), (
                f"Retrieved context failed cross-check: low semantic relevance ({best_score:.2f}). "
                f"Missing key terms: {query_terms}."
            )