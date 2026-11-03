"""
Hermetic tests: chunking (incl. syntax-error fallback), chunk ids, and
retriever + reranker wiring using fake embedder / vector store.
"""
import tempfile
from pathlib import Path

import src.ingestion.embedder as embedder_module
from src.ingestion.chunker import CodebaseChunker
from src.retrieval.reranker import CodeReranker
from src.retrieval.retriever import CodeRetriever

GOOD_PY = '''
def add(a, b):
    return a + b


class Calc:
    def mul(self, a, b):
        return a * b
'''

BROKEN_PY = '''
def broken(x:
    return x +
'''


def _write(dirpath, name, text):
    path = Path(dirpath) / name
    path.write_text(text, encoding="utf-8")
    return path


def test_chunk_id_exists_is_unique_and_stable():
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "m.py", GOOD_PY)
        first = CodebaseChunker().chunk_file(path)
        second = CodebaseChunker().chunk_file(path)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids)) >= 3
    assert ids == [c.chunk_id for c in second]


def test_chunk_id_changes_when_code_changes():
    with tempfile.TemporaryDirectory() as d:
        a = CodebaseChunker().chunk_file(_write(d, "a.py", GOOD_PY))
        b = CodebaseChunker().chunk_file(_write(d, "a.py", GOOD_PY.replace("a + b", "a - b")))
    assert a[0].chunk_id != b[0].chunk_id


def test_syntax_error_file_is_indexed_via_fallback_not_skipped():
    with tempfile.TemporaryDirectory() as d:
        broken = _write(d, "broken.py", BROKEN_PY)
        chunks = CodebaseChunker().chunk_codebase([broken])
    assert chunks, "files with syntax errors must still be indexed"
    assert chunks[0].chunk_type == "text_window"
    assert "def broken" in chunks[0].code
    assert chunks[0].chunk_id


def test_non_python_files_get_language_tagged_windows():
    js = "\n".join(f"const x{i} = {i};" for i in range(130))
    with tempfile.TemporaryDirectory() as d:
        chunks = CodebaseChunker().chunk_codebase([_write(d, "app.js", js)])
    assert len(chunks) >= 2
    assert {c.language for c in chunks} == {"javascript"}
    assert chunks[0].start_line == 1 and chunks[1].start_line < chunks[0].end_line  # overlap


# ------------------------------ retriever + reranker ------------------------------
class FakeEmbedder:
    def embed_text(self, text):
        return [0.0]


class FakeStore:
    """Returns candidates in a fixed order; records the top_k it was asked for."""

    def __init__(self):
        self.asked_top_k = None

    def search(self, query_embedding, top_k=5):
        self.asked_top_k = top_k
        docs = [
            ("id_noise1", "def helper(): pass", "helper", 0.20),
            ("id_noise2", "def other(): pass", "other", 0.25),
            ("id_noise3", "def misc(): pass", "misc", 0.30),
            ("id_target", "def redact_secrets(text): ...", "redact_secrets", 0.35),
        ]
        docs = docs[:top_k]
        return {
            "ids": [[d[0] for d in docs]],
            "documents": [[d[1] for d in docs]],
            "metadatas": [[{"name": d[2], "qualified_name": d[2], "file_path": "x.py",
                            "start_line": 1, "end_line": 2} for d in docs]],
            "distances": [[d[3] for d in docs]],
        }


def _retriever(**kw):
    store = FakeStore()
    return CodeRetriever(embedder=FakeEmbedder(), vector_store=store, reranker=CodeReranker(), **kw), store


def test_reranker_is_wired_in_and_oversamples():
    retriever, store = _retriever(use_reranker=True, oversample=4)
    results = retriever.search("where is redact_secrets defined", top_k=1)
    assert store.asked_top_k == 4                      # asked for top_k * oversample candidates
    assert results["ids"][0] == ["id_target"]          # lexical/symbol match promoted to #1
    assert "rerank_scores" in results


def test_rerank_can_be_disabled_per_call():
    retriever, store = _retriever(use_reranker=True, oversample=4)
    results = retriever.search("where is redact_secrets defined", top_k=1, rerank=False)
    assert store.asked_top_k == 1
    assert results["ids"][0] == ["id_noise1"]          # raw vector order
    assert "rerank_scores" not in results


def test_shared_embedder_loads_the_model_once():
    created = []

    class Counting:
        def __init__(self):
            created.append(1)

    original, embedder_module.CodeEmbedder = embedder_module.CodeEmbedder, Counting
    embedder_module._shared_embedder = None
    try:
        a = embedder_module.get_shared_embedder()
        b = embedder_module.get_shared_embedder()
    finally:
        embedder_module.CodeEmbedder = original
        embedder_module._shared_embedder = None
    assert a is b and len(created) == 1
