"""
Keep `pytest` fast and hermetic.

Many files in this folder are print-style integration scripts: they run at import time and
need API keys, an embedding model, a vector store, or LangGraph. They are skipped by default.

    pytest                      # unit tests only (what CI runs)
    RUN_INTEGRATION=1 pytest    # also collect the integration scripts (needs the full stack)
"""
import ast
import os
from pathlib import Path

_HERE = Path(__file__).parent
_HEAVY_IMPORTS = (
    "langgraph", "chromadb", "sentence_transformers", "groq", "ollama", "langchain",
    "src.agent.graph", "src.agent.nodes", "src.evaluation", "app.main",
)


def _is_unit_test_module(path: Path) -> bool:
    """True if the file defines real test functions/classes and imports no heavyweight stack."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    has_tests = any(
        isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name.lower().startswith("test")
        for n in tree.body
    )
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    heavy = any(i == h or i.startswith(h + ".") for i in imported for h in _HEAVY_IMPORTS)
    return has_tests and not heavy


def _ignored_files() -> list:
    if os.getenv("RUN_INTEGRATION") == "1":
        return []
    return sorted(p.name for p in _HERE.glob("test_*.py") if not _is_unit_test_module(p))


collect_ignore = _ignored_files()
