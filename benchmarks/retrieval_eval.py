"""
Retrieval quality: does vector search find the right FILE, and does reranking help?

    python -m benchmarks.retrieval_eval --k 5

Requires an index of THIS repository (POST /index with target_path "." first).
Reports hit@1, hit@k and MRR for vector-only vs vector + reranker on
data/retrieval_eval.json (ground truth is validated against the real files by the test suite).
Small dataset => treat differences of a few points as noise.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "retrieval_eval.json"


def _norm(path: str) -> str:
    return str(path).replace("\\", "/")


def result_files(results: dict) -> List[str]:
    """Distinct file paths of a vector-store result, in rank order."""
    seen: List[str] = []
    for meta in (results.get("metadatas") or [[]])[0]:
        path = _norm(meta.get("file_path", ""))
        if path and path not in seen:
            seen.append(path)
    return seen


def first_relevant_rank(files: List[str], relevant: List[str]) -> Optional[int]:
    """1-based rank of the first retrieved file matching any relevant path (suffix match)."""
    wanted = [_norm(r) for r in relevant]
    for rank, path in enumerate(files, 1):
        if any(path == w or path.endswith("/" + w) for w in wanted):
            return rank
    return None


def evaluate(search: Callable[[str], dict], dataset: List[dict], k: int) -> Dict[str, float]:
    ranks = [first_relevant_rank(result_files(search(item["question"])), item["relevant_files"]) for item in dataset]
    n = len(ranks)
    return {
        "questions": n,
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        f"hit@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    from src.retrieval.retriever import CodeRetriever
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    retriever = CodeRetriever()
    rows = {
        "vector only": evaluate(lambda q: retriever.search(q, top_k=args.k, rerank=False), dataset, args.k),
        "vector + reranker": evaluate(lambda q: retriever.search(q, top_k=args.k, rerank=True), dataset, args.k),
    }
    k = args.k
    print(f"\n{'config':<20}{'hit@1':>8}{f'hit@{k}':>9}{'MRR':>8}   (n={len(dataset)})")
    for name, m in rows.items():
        print(f"{name:<20}{m['hit@1']:>8.2f}{m[f'hit@{k}']:>9.2f}{m['mrr']:>8.2f}")
    if not args.no_save:
        out = ROOT / "benchmarks" / "results"
        out.mkdir(exist_ok=True)
        path = out / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps({"k": k, "results": rows}, indent=2), encoding="utf-8")
        print(f"Saved to {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
