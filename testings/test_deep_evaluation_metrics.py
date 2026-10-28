import sys
import json
from pathlib import Path

# Setup sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.evaluation import (
    RagasEvaluator,
    compute_bleu,
    compute_rouge,
    compute_meteor,
    compute_all_lexical_metrics,
    compute_semantic_similarity,
)
from src.config import settings


def test_deep_metrics():
    print("=" * 80)
    print("TESTING DEEP RAG & NLP EVALUATION MATRICES (3 DEPENDENT CATEGORIES)")
    print("=" * 80)
    print(f"Model Provider: {settings.MODEL_PROVIDER} ({settings.GROQ_MODEL if settings.MODEL_PROVIDER == 'groq' else settings.OLLAMA_MODEL})")
    print("=" * 80)

    evaluator = RagasEvaluator()

    question = "Where is CodebaseChunker defined and how does it chunk functions in src/ingestion/chunker.py?"

    ground_truth = (
        "CodebaseChunker is defined in src/ingestion/chunker.py spanning lines 19-161. "
        "It chunks functions via chunk_file and an internal recursive _traverse function over ast.AST nodes. "
        "For ast.FunctionDef and ast.AsyncFunctionDef, it extracts lines using decorator start boundaries, "
        "slices code text, and stores metadata in a CodeChunk dataclass."
    )

    retrieved_contexts = [
        """File: src/ingestion/chunker.py
import ast

class CodebaseChunker:
    # Lines 19-161 in src/ingestion/chunker.py
    def chunk_file(self, file_path: str):
        # Recursively traverses AST nodes to extract code chunks
        def _traverse(nodes):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                    # Creates CodeChunk with start and end line metadata
                    return CodeChunk(name=node.name, start_line=start, end_line=node.end_lineno)
        """,
        """File: src/retrieval/vector_store.py
class CodeVectorStore:
    def add_chunks(self, ids, embeddings, documents, metadatas):
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=documents)
        """
    ]

    candidate_answer = (
        "CodebaseChunker is located in `src/ingestion/chunker.py` (lines 19-161). "
        "It processes Python functions by recursively traversing the AST tree with a `_traverse` helper. "
        "When encountering `ast.FunctionDef` or `ast.AsyncFunctionDef`, it takes decorator positions into account "
        "and creates a `CodeChunk` object with start and end line metadata."
    )

    print("\n--- 1. CONTEXT-DEPENDENT METRICS (Reference-Free) ---")
    faithfulness, claims = evaluator.compute_faithfulness(question, candidate_answer, retrieved_contexts)
    utilization, util_details = evaluator.compute_context_utilization(candidate_answer, retrieved_contexts)
    print(f"  [+] Faithfulness Score:     {faithfulness:.2f} ({sum(1 for c in claims if c.supported)}/{len(claims)} claims grounded)")
    print(f"  [+] Context Utilization:    {utilization:.2f} ({util_details['utilized_chunks']}/{util_details['total_chunks']} chunks used in answer)")
    assert faithfulness >= 0.50, f"Faithfulness should be >= 0.50, got {faithfulness}"


    print("\n--- 2. GROUND TRUTH-DEPENDENT METRICS (Reference-Based) ---")
    print("A. Advanced LLM-as-a-Judge Reference Metrics:")
    correctness, judge_metrics = evaluator.compute_answer_correctness(question, candidate_answer, ground_truth)
    semantic_sim = compute_semantic_similarity(ground_truth, candidate_answer)
    print(f"  [+] Answer Correctness:     {correctness:.2f}")
    print(f"      - Factual F1:           {judge_metrics.factual_correctness_f1:.2f} (TP={judge_metrics.true_positives}, FP={judge_metrics.false_positives}, FN={judge_metrics.false_negatives})")
    print(f"      - Semantic Similarity:  {semantic_sim:.2f}")
    assert correctness >= 0.60, f"Correctness should be >= 0.60, got {correctness}"

    print("\nB. Classic Lexical / Statistical Metrics:")
    lexical = compute_all_lexical_metrics(ground_truth, candidate_answer)
    print(f"  [+] BLEU (Cumulative):      {lexical['bleu']:.4f} (BP={lexical['brevity_penalty']:.2f}, BLEU-1={lexical['bleu_1']:.2f}, BLEU-2={lexical['bleu_2']:.2f})")
    print(f"  [+] ROUGE-1 F1:             {lexical['rouge_1']['f1']:.4f} (P={lexical['rouge_1']['precision']:.2f}, R={lexical['rouge_1']['recall']:.2f})")
    print(f"  [+] ROUGE-2 F1:             {lexical['rouge_2']['f1']:.4f} (P={lexical['rouge_2']['precision']:.2f}, R={lexical['rouge_2']['recall']:.2f})")
    print(f"  [+] ROUGE-L F1:             {lexical['rouge_l']['f1']:.4f} (P={lexical['rouge_l']['precision']:.2f}, R={lexical['rouge_l']['recall']:.2f})")
    print(f"  [+] METEOR Score:           {lexical['meteor']:.4f}")
    assert lexical["rouge_1"]["f1"] > 0.30, "ROUGE-1 F1 should be > 0.30"
    assert lexical["meteor"] > 0.30, "METEOR should be > 0.30"

    print("\n--- 3. RETRIEVAL-DEPENDENT METRICS ---")
    recall, recall_details = evaluator.compute_context_recall(ground_truth, retrieved_contexts)
    precision, prec_feedback = evaluator.compute_context_precision(question, retrieved_contexts, ground_truth=ground_truth)
    print(f"  [+] Context Recall:         {recall:.2f} ({recall_details['supported_gt_claims']}/{recall_details['total_gt_claims']} GT claims covered by retrieved context)")
    print(f"  [+] Context Precision:      {precision:.2f}")
    assert recall >= 0.50, f"Recall should be >= 0.50, got {recall}"

    print("\n--- 4. FULL DEEP EVALUATION REPORT (Unified Pipeline) ---")
    report = evaluator.evaluate_deep(
        question=question,
        answer=candidate_answer,
        contexts=retrieved_contexts,
        ground_truth=ground_truth
    )
    print(f"Composite Score:              {report.composite_score:.2f} / 1.00")
    print(f"Scorecard Summary:            {report.summary}")

    print("\n" + "=" * 80)
    print("ALL 3 DEPENDENT EVALUATION CATEGORIES PASSED WITH FLYING COLORS!")
    print("=" * 80)


if __name__ == "__main__":
    test_deep_metrics()
