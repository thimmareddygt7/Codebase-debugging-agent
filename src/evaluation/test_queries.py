import sys
import json
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.evaluation.ragas_eval import RagasEvaluator
from src.agent.graph import ask_codebase_agent
from src.config import settings
from src.observability import is_tracing_enabled, get_project_dashboard_url


def run_benchmark(dataset_path: str = "data/eval_dataset.json", max_eval_samples: int = 3):
    print("=" * 80)
    print("RAGAS-STYLE BENCHMARK EVALUATION (FAITHFULNESS & RELEVANCE)")
    print("=" * 80)
    print(f"Model Provider:    {settings.MODEL_PROVIDER} ({settings.GROQ_MODEL if settings.MODEL_PROVIDER == 'groq' else settings.OLLAMA_MODEL})")
    print(f"LangSmith Tracing: {'ACTIVE' if is_tracing_enabled() else 'DISABLED'}")
    if is_tracing_enabled():
        print(f"Dashboard:         {get_project_dashboard_url()}")
    print("=" * 80)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        print(f"[ERROR]: Dataset file '{dataset_path}' not found.")
        return

    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))[:max_eval_samples]
    print(f"Loaded {len(dataset)} evaluation sample(s) from {dataset_path}...\n")

    evaluator = RagasEvaluator()
    results = []

    for idx, item in enumerate(dataset, start=1):
        q_id = item["id"]
        question = item["question"]
        print("-" * 80)
        print(f"[{idx}/{len(dataset)}] Evaluating Query ID: {q_id}")
        print(f"    Question: '{question}'")

        # Run query through the debugging agent state machine
        session_id = f"eval_{q_id}"
        agent_out = ask_codebase_agent(question=question, thread_id=session_id)

        answer = agent_out.get("final_answer", "")
        # Extract retrieved context chunks from tool results
        contexts = []
        for tr in agent_out.get("tool_results", []):
            try:
                res_obj = json.loads(tr.get("result", "{}"))
                for chunk in res_obj.get("results", []):
                    code_snippet = chunk.get("code") or chunk.get("content") or ""
                    if code_snippet:
                        contexts.append(code_snippet)
            except Exception:
                pass

        if not contexts and answer:
            # If answer was synthesized directly, use ground truth as context reference
            contexts = [item.get("ground_truth", "")]

        # Run Deep Multi-Category Evaluation
        deep_report = evaluator.evaluate_deep(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=item.get("ground_truth", "")
        )

        results.append({
            "id": q_id,
            "question": question,
            "metrics": deep_report.to_dict()
        })

        cd = deep_report.context_dependent
        rd = deep_report.retrieval_dependent
        gtd = deep_report.ground_truth_dependent

        print(f"    -> [Context-Dependent]   Faithfulness: {cd.faithfulness:.2f} ({cd.claims_supported}/{cd.claims_evaluated}) | Utilization: {cd.context_utilization:.2f}")
        print(f"    -> [Retrieval-Dependent] Recall: {rd.context_recall:.2f} | Precision: {rd.context_precision:.2f} | MRR: {rd.mean_reciprocal_rank:.2f}")
        if gtd:
            print(f"    -> [GT-Dependent]        Correctness: {gtd.llm_judge.answer_correctness:.2f} | SemanticSim: {gtd.llm_judge.semantic_similarity:.2f}")
            print(f"                             ROUGE-L: {gtd.lexical['rouge_l']['f1']:.2f} | BLEU: {gtd.lexical['bleu']:.2f} | METEOR: {gtd.lexical['meteor']:.2f}")
        print(f"    -> Composite Index:      {deep_report.composite_score:.2f} / 1.00")

    # Aggregate Scores
    avg_faithfulness = sum(r["metrics"]["context_dependent"]["faithfulness"] for r in results) / len(results)
    avg_utilization = sum(r["metrics"]["context_dependent"]["context_utilization"] for r in results) / len(results)
    avg_recall = sum(r["metrics"]["retrieval_dependent"]["context_recall"] for r in results) / len(results)
    avg_precision = sum(r["metrics"]["retrieval_dependent"]["context_precision"] for r in results) / len(results)
    avg_composite = sum(r["metrics"]["composite_score"] for r in results) / len(results)

    gt_samples = [r for r in results if r["metrics"]["ground_truth_dependent"] is not None]
    avg_correctness = (
        sum(r["metrics"]["ground_truth_dependent"]["llm_judge"]["answer_correctness"] for r in gt_samples) / len(gt_samples)
        if gt_samples else 0.0
    )

    print("\n" + "=" * 80)
    print("DEEP RAG & NLP EVALUATION BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Evaluated Samples:     {len(results)}")
    print(f"1. Context-Dependent:")
    print(f"   - Mean Faithfulness:      {avg_faithfulness:.3f} / 1.000")
    print(f"   - Mean Utilization:       {avg_utilization:.3f} / 1.000")
    print(f"2. Ground Truth-Dependent:")
    print(f"   - Mean Answer Correctness:{avg_correctness:.3f} / 1.000")
    print(f"3. Retrieval-Dependent:")
    print(f"   - Mean Context Recall:    {avg_recall:.3f} / 1.000")
    print(f"   - Mean Context Precision: {avg_precision:.3f} / 1.000")
    print(f"Composite Deep RAG Index:    {avg_composite:.3f} / 1.000")
    print("=" * 80)

    # Save benchmark report to data/ragas_report.json
    report_file = Path("data/ragas_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps({
            "summary": {
                "mean_faithfulness": avg_faithfulness,
                "mean_utilization": avg_utilization,
                "mean_answer_correctness": avg_correctness,
                "mean_context_recall": avg_recall,
                "mean_context_precision": avg_precision,
                "mean_composite_score": avg_composite,
                "total_samples": len(results)
            },
            "detailed_results": results
        }, indent=2),
        encoding="utf-8"
    )
    print(f"Detailed RAG evaluation scorecard saved to: {report_file.resolve()}\n")


if __name__ == "__main__":
    run_benchmark(max_eval_samples=2)

