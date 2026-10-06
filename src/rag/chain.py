from src.retrieval.retriever import CodeRetriever
from src.rag.context_builder import CodeContextBuilder
from src.rag.llm import OllamaLLM
from src.rag.citations import CitationBuilder


class CodeRAGChain:

    def __init__(self):

        print("Initializing Code RAG chain...")

        self.retriever = CodeRetriever()
        self.context_builder = CodeContextBuilder()
        self.llm = OllamaLLM()
        self.citation_builder = CitationBuilder()

        print("Code RAG chain initialized.")

    def build_prompt(
        self,
        question: str,
        context: str
    ) -> str:

        prompt = f"""
You are an AI assistant that answers questions about a software codebase.

Use ONLY the provided code context to answer the user's question.

If the answer cannot be determined from the provided context, say:
"I cannot determine the answer from the retrieved code."

Do not invent code, files, functions, or explanations that are not supported by the context.

User Question:
{question}

Retrieved Code Context:
{context}

Answer:
"""

        return prompt

    def ask(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:

        results = self.retriever.search(
            query=question,
            top_k=top_k
        )

        context = self.context_builder.build_context(
            results
        )

        prompt = self.build_prompt(
            question=question,
            context=context
        )

        answer = self.llm.generate(
            prompt
        )

        citations = self.citation_builder.build_citations(
            results
        )

        return {
            "answer": answer,
            "citations": citations,
        }