"""
Retrieval-Augmented Generation (RAG) components: chains, citations, CRAG, and Self-RAG.
"""
from src.rag.chain import CodeRAGChain
from src.rag.citations import CitationBuilder, CitationFormatter
from src.rag.citation_verifier import CitationVerifier
from src.rag.context_builder import ContextBuilder
from src.rag.crag import CorrectiveRAG
from src.rag.self_rag import SelfRAGVerifier
from src.rag.llm import GroqLLM, OllamaLLM, get_llm, LLMFactory

__all__ = [
    "CodeRAGChain",
    "CitationBuilder",
    "CitationFormatter",
    "CitationVerifier",
    "ContextBuilder",
    "CorrectiveRAG",
    "SelfRAGVerifier",
    "GroqLLM",
    "OllamaLLM",
    "get_llm",
    "LLMFactory",
]
