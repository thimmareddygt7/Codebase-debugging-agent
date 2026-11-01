import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.evaluation import RagasEvaluator
from src.config import settings


def main():
    print("=" * 80)
    print("RAGAS FAITHFULNESS & RELEVANCE SCORING TEST")
    print("=" * 80)
    print(f"Engine: {settings.MODEL_PROVIDER} ({settings.GROQ_MODEL if settings.MODEL_PROVIDER == 'groq' else settings.OLLAMA_MODEL})")
    print("=" * 80)

    evaluator = RagasEvaluator()

    question = "Where is CodebaseChunker defined and what AST nodes does it parse for functions?"

    grounded_context = [
        """File: src/ingestion/chunker.py
class CodebaseChunker:
    # Lines 19-161
    def chunk_file(self, file_path):
        def _traverse(nodes):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # creates CodeChunk for function
                    return CodeChunk(...)
        """
    ]

    # Test 1: High faithfulness, high relevance answer
    good_answer = (
        "CodebaseChunker is defined in `src/ingestion/chunker.py` (lines 19-161). "
        "It parses functions by traversing AST nodes and checking for `ast.FunctionDef` "
        "and `ast.AsyncFunctionDef` instances inside the `_traverse` helper function."
    )

    print("\n--- Test 1: Grounded Answer ---")
    metrics_good = evaluator.evaluate(
        question=question,
        answer=good_answer,
        contexts=grounded_context
    )
    print(f"Faithfulness Score:    {metrics_good.faithfulness:.2f}")
    print(f"Answer Relevance:      {metrics_good.answer_relevance:.2f}")
    print(f"Context Precision:     {metrics_good.context_precision:.2f}")
    print(f"Composite RAG Quality: {metrics_good.composite_score:.2f} / 1.00")
    print(f"Claims Evaluated:      {metrics_good.claims_supported}/{metrics_good.claims_evaluated} supported")
    assert metrics_good.faithfulness >= 0.70, f"Expected high faithfulness, got {metrics_good.faithfulness}"
    print("[PASS] Grounded answer achieved high faithfulness.")

    # Test 2: Hallucinated / ungrounded claims
    hallucinated_answer = (
        "CodebaseChunker is a Rust microservice running in AWS Lambda that communicates "
        "via gRPC protocol and uses regular expressions instead of AST trees to parse functions."
    )

    print("\n--- Test 2: Hallucinated Answer ---")
    metrics_bad = evaluator.evaluate(
        question=question,
        answer=hallucinated_answer,
        contexts=grounded_context
    )
    print(f"Faithfulness Score:    {metrics_bad.faithfulness:.2f}")
    print(f"Answer Relevance:      {metrics_bad.answer_relevance:.2f}")
    print(f"Context Precision:     {metrics_bad.context_precision:.2f}")
    print(f"Composite RAG Quality: {metrics_bad.composite_score:.2f} / 1.00")
    print(f"Claims Evaluated:      {metrics_bad.claims_supported}/{metrics_bad.claims_evaluated} supported")
    assert metrics_bad.faithfulness < metrics_good.faithfulness, "Hallucinated answer should have lower faithfulness!"
    print("[PASS] Hallucinated claims correctly detected with low faithfulness score.")

    print("\n" + "=" * 80)
    print("ALL RAGAS SCORING TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
