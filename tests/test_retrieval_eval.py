"""Retrieval-eval metrics and dataset integrity (ground truth must match the real code)."""
import json
from pathlib import Path

from benchmarks.retrieval_eval import DATASET, evaluate, first_relevant_rank, result_files

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads(DATASET.read_text(encoding="utf-8"))


def _results(*paths):
    return {"metadatas": [[{"file_path": p} for p in paths]]}


def test_dataset_ground_truth_matches_the_real_repository():
    assert len(DATA) >= 15 and len({d["id"] for d in DATA}) == len(DATA)
    for item in DATA:
        assert item["relevant_files"], item["id"]
        texts = []
        for rel in item["relevant_files"]:
            path = ROOT / rel
            assert path.is_file(), f"{item['id']}: {rel} does not exist"
            texts.append(path.read_text(encoding="utf-8"))
        assert any(item["expected_symbol"] in t for t in texts), f"{item['id']}: symbol missing"


def test_rank_matching_handles_absolute_paths_and_windows_separators():
    files = result_files(_results("C:\\repo\\src\\a.py", "/srv/app/src/guardrails/secrets.py", "src/a.py"))
    assert files == ["C:/repo/src/a.py", "/srv/app/src/guardrails/secrets.py", "src/a.py"]
    assert first_relevant_rank(files, ["src/guardrails/secrets.py"]) == 2
    assert first_relevant_rank(files, ["src/a.py"]) == 1
    assert first_relevant_rank(files, ["not/there.py"]) is None
    assert first_relevant_rank(["src/xsecrets.py"], ["secrets.py"]) is None   # no partial-name matches


def test_chunks_from_the_same_file_do_not_inflate_ranks():
    assert result_files(_results("a.py", "a.py", "a.py", "b.py")) == ["a.py", "b.py"]


def test_metrics_perfect_partial_and_zero():
    data = [{"question": "q1", "relevant_files": ["a.py"]}, {"question": "q2", "relevant_files": ["b.py"]}]
    perfect = evaluate(lambda q: _results("a.py") if q == "q1" else _results("b.py"), data, k=3)
    assert perfect == {"questions": 2, "hit@1": 1.0, "hit@3": 1.0, "mrr": 1.0}
    partial = evaluate(lambda q: _results("x.py", "a.py") if q == "q1" else _results("z.py"), data, k=3)
    assert partial["hit@1"] == 0.0 and partial["hit@3"] == 0.5 and partial["mrr"] == 0.25
    assert evaluate(lambda q: _results("nope.py"), data, k=3)["mrr"] == 0.0
