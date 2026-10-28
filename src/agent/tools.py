from pathlib import Path
from src.retrieval.retriever import CodeRetriever
from src.guardrails import FileSystemGuardrail, redact_secrets

fs_guard = FileSystemGuardrail()


class CodeSearchTool:
    """Tool to search the indexed codebase for relevant source code."""

    NAME = "code_search"
    DESCRIPTION = (
        "Search the indexed codebase for relevant source code. "
        "Use this tool when you need to find classes, functions, "
        "methods, files, or implementation details in the codebase."
    )

    def __init__(self, persist_directory: str = "vector_store"):
        self.retriever = CodeRetriever(persist_directory=persist_directory)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.retriever.search(query=query, top_k=top_k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "file_path": metadatas[i]["file_path"],
                "chunk_name": metadatas[i]["name"],
                "qualified_name": metadatas[i]["qualified_name"],
                "start_line": metadatas[i]["start_line"],
                "end_line": metadatas[i]["end_line"],
                "distance": distances[i],
                "code": redact_secrets(documents[i]),
            }
            for i in range(len(documents))
        ]

    def execute(self, query: str, top_k: int = 5) -> list[dict]:
        return self.search(query=query, top_k=top_k)

    def get_tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search question or description "
                                "of the code you want to find."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of relevant code chunks to retrieve.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


class DocSearchTool:
    """Tool to search documentation, markdown files, and config instructions."""

    NAME = "doc_search"
    DESCRIPTION = (
        "Search documentation, README files, guides, and environment configs. "
        "Use this tool when handling conceptual queries, installation issues, "
        "environment setup errors, or documentation questions."
    )

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.doc_extensions = {".md", ".rst", ".txt", ".json", ".example"}

    def search(self, query: str) -> list[dict]:
        """Performs safe keyword-based retrieval across documentation files."""
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        matches = []

        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue

            # Filesystem Guardrail: Block protected paths (.git, credentials, live .env)
            is_safe, _ = fs_guard.is_safe_read_path(path)
            if not is_safe:
                continue

            # Block live .env strictly (only allow .env.example)
            if path.name == ".env" or path.name.startswith(".env.") and not path.name.endswith(".example"):
                continue

            if any(part in {".git", ".venv", "venv", "__pycache__", "vector_store"} for part in path.parts):
                continue

            if path.suffix.lower() in self.doc_extensions:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    lines = text.splitlines()
                    for idx, line in enumerate(lines, start=1):
                        line_lower = line.lower()
                        if any(term in line_lower for term in query_terms):
                            snippet_start = max(0, idx - 2)
                            snippet_end = min(len(lines), idx + 3)
                            snippet = "\n".join(lines[snippet_start:snippet_end])
                            matches.append({
                                "file_path": str(path),
                                "line": idx,
                                "content": redact_secrets(snippet)
                            })
                            if len(matches) >= 5:
                                return matches
                except Exception:
                    continue

        return matches

    def execute(self, query: str) -> list[dict]:
        return self.search(query)

    def get_tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search term or error concept to search docs for."
                        }
                    },
                    "required": ["query"]
                }
            }
        }