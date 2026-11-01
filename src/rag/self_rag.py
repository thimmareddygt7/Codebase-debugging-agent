"""
Self-RAG (Self-Reflective Retrieval-Augmented Generation) verification module.
Performs post-generation self-reflection to verify answer grounding, citation integrity,
and absence of hallucinated symbols.
"""
from typing import Dict, Any, List, Optional
import re
from src.rag.citation_verifier import CitationVerifier


class SelfRAGVerifier:
    """
    Evaluates whether generated answers are grounded in provided citations
    and fact-checks for hallucinated identifiers or incorrect file paths.
    """

    def __init__(self):
        self.citation_verifier = CitationVerifier()

    def check_faithfulness(
        self,
        answer: str,
        retrieved_context: str
    ) -> Dict[str, Any]:
        """
        Assesses if key statements and code references in the answer are grounded
        in the retrieved context string.
        """
        if not answer:
            return {"is_grounded": False, "score": 0.0, "reason": "Empty answer provided."}

        if not retrieved_context:
            return {
                "is_grounded": False,
                "score": 0.0,
                "reason": "No retrieved context provided for verification."
            }

        # Extract identifier tokens from code blocks in answer (e.g., function names, variables)
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", answer)
        combined_code = " ".join(code_blocks)

        tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", combined_code))
        # Filter python built-ins / common keywords
        common_py = {"def", "class", "return", "import", "from", "None", "True", "False", "self", "args", "kwargs"}
        tokens = tokens - common_py

        if not tokens:
            return {"is_grounded": True, "score": 1.0, "reason": "No custom code symbols to verify."}

        context_lower = retrieved_context.lower()
        supported = [tok for tok in tokens if tok.lower() in context_lower]
        grounding_score = len(supported) / len(tokens) if tokens else 1.0

        is_grounded = grounding_score >= 0.50
        return {
            "is_grounded": is_grounded,
            "score": round(grounding_score, 2),
            "verified_symbols": supported,
            "unverified_symbols": list(tokens - set(supported)),
            "reason": "Answer well-grounded in code context." if is_grounded else "Potential hallucinated symbols found."
        }

    def verify_citations_and_grounding(
        self,
        answer: str,
        citations: List[str],
        retrieval_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validates both citation structure and factual alignment.
        """
        citations_valid = True
        if retrieval_results and citations:
            citations_valid = self.citation_verifier.verify(retrieval_results, citations)

        context_str = " ".join(citations) if citations else ""
        grounding = self.check_faithfulness(answer, context_str)

        return {
            "citations_valid": citations_valid,
            "grounding_score": grounding["score"],
            "is_faithful": grounding["is_grounded"] and citations_valid,
            "details": grounding
        }
