import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from src.rag.llm import get_llm
from src.observability import trace_run, get_langsmith_client, is_tracing_enabled
from src.guardrails.secrets import redact_secrets
from src.evaluation.lexical_metrics import compute_all_lexical_metrics
from src.evaluation.semantic_metrics import compute_semantic_similarity


@dataclass
class ClaimVerification:
    statement: str
    supported: bool
    evidence: str


@dataclass
class RagasMetrics:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    composite_score: float
    claims_evaluated: int
    claims_supported: int
    claims_detail: List[Dict[str, Any]]
    context_chunks_evaluated: int
    context_chunks_relevant: int
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------
# Detailed Deep Evaluation Data Classes
# -------------------------------------------------------------
@dataclass
class ContextDependentMetrics:
    """1. Context-Dependent Metrics (Reference-Free / System Evaluation)"""
    faithfulness: float
    context_utilization: float
    claims_evaluated: int
    claims_supported: int
    utilized_chunks: int
    total_chunks: int
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMJudgeMetrics:
    """Advanced LLM-as-a-Judge Reference Metrics"""
    answer_correctness: float
    factual_correctness_f1: float
    semantic_similarity: float
    true_positives: int
    false_positives: int
    false_negatives: int
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundTruthDependentMetrics:
    """2. Ground Truth-Dependent Metrics (Reference-Based Evaluation)"""
    llm_judge: LLMJudgeMetrics
    lexical: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalDependentMetrics:
    """3. Retrieval-Dependent Metrics"""
    context_recall: float
    context_precision: float
    mean_reciprocal_rank: float
    total_chunks: int
    relevant_chunks: int
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeepEvaluationReport:
    """Comprehensive multi-category RAG & NLP evaluation scorecard."""
    context_dependent: ContextDependentMetrics
    ground_truth_dependent: Optional[GroundTruthDependentMetrics]
    retrieval_dependent: RetrievalDependentMetrics
    composite_score: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RagasEvaluator:
    """
    Comprehensive RAG & NLP Evaluation Engine:
    
    1. Context-Dependent Metrics (Reference-Free / System Evaluation):
       - Faithfulness (Anti-Hallucination Claim Verification)
       - Context Utilization / Grader (Fraction of retrieved chunks utilized in answer)
       
    2. Ground Truth-Dependent Metrics (Reference-Based Evaluation):
       - A. Advanced LLM-as-a-Judge:
            * Answer Correctness (Factual TP/FP/FN + Semantic Similarity)
            * Semantic Similarity (Embedding Cosine Similarity)
       - B. Classic Lexical / Statistical Metrics:
            * ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
            * BLEU (BLEU-1 to BLEU-4 with Brevity Penalty)
            * METEOR (Harmonic mean with chunk fragmentation penalty)
            
    3. Retrieval-Dependent Metrics:
       - Context Recall (Fraction of Ground Truth claims covered by retrieved chunks)
       - Context Precision & MRR (Rank-aware signal-to-noise ratio)
    """

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def _extract_json(self, raw_output: str) -> Any:
        """Extracts JSON structure from markdown or text output."""
        try:
            return json.loads(raw_output.strip())
        except Exception:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw_output)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except Exception:
                pass

        obj_match = re.search(r"\{[\s\S]*\}", raw_output)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except Exception:
                pass

        return None

    # =============================================================
    # CATEGORY 1: Context-Dependent Metrics (Reference-Free)
    # =============================================================

    @trace_run(name="ragas_eval_faithfulness", run_type="chain", tags=["ragas", "faithfulness"])
    def compute_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Tuple[float, List[ClaimVerification]]:
        """
        Calculates Faithfulness:
        Ratio of atomic factual claims in the answer that can be directly
        inferred from the retrieved context chunks (supported / total).
        """
        if not answer or not answer.strip():
            return 0.0, []

        if not contexts:
            return 0.0, []

        # Step 1: Decompose answer into discrete claims
        prompt_extract = (
            "You are a rigorous code verification evaluator. Decompose the following answer "
            "into a JSON list of discrete, atomic factual statements/claims about the codebase.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Format your output strictly as a JSON list of strings, for example:\n"
            '["CodebaseChunker is defined in src/ingestion/chunker.py", "It parses functions with ast.FunctionDef"]'
        )

        raw_claims = self.llm.generate(prompt_extract)
        parsed_claims = self._extract_json(raw_claims)

        if not parsed_claims or not isinstance(parsed_claims, list):
            parsed_claims = [
                s.strip() for s in re.split(r"[.\n]+", answer)
                if len(s.strip()) > 20 and not s.strip().startswith(("#", "-", "*", "`"))
            ][:5]

        if not parsed_claims:
            return 1.0, []

        parsed_claims = parsed_claims[:5]
        combined_context = "\n---\n".join([c[:750] for c in contexts[:4]])

        # Step 2: Verify each claim against retrieved context
        prompt_verify = (
            "You are a strict code grounding critic. For each claim below, determine whether it is "
            "directly supported and verified by the provided Code Context.\n\n"
            f"Code Context:\n{combined_context}\n\n"
            f"Claims to verify:\n{json.dumps(parsed_claims, indent=2)}\n\n"
            "Return a JSON list of objects with keys 'statement', 'supported' (boolean), and 'evidence' (short quote or reason).\n"
            "Output JSON only:"
        )

        raw_verifications = self.llm.generate(prompt_verify)
        parsed_verifications = self._extract_json(raw_verifications)

        claim_results: List[ClaimVerification] = []

        if parsed_verifications and isinstance(parsed_verifications, list):
            for item in parsed_verifications:
                if isinstance(item, dict):
                    claim_results.append(
                        ClaimVerification(
                            statement=item.get("statement") or item.get("claim", ""),
                            supported=bool(item.get("supported", False)),
                            evidence=item.get("evidence", "")
                        )
                    )

        if not claim_results:
            for claim in parsed_claims:
                words = [w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3]
                matches = sum(1 for w in words if w in combined_context.lower())
                is_supported = (matches / max(len(words), 1)) >= 0.5
                claim_results.append(
                    ClaimVerification(
                        statement=claim,
                        supported=is_supported,
                        evidence="Lexical context overlap" if is_supported else "Insufficient context overlap"
                    )
                )

        supported_count = sum(1 for c in claim_results if c.supported)
        total_count = max(len(claim_results), 1)
        score = round(supported_count / total_count, 3)

        return score, claim_results

    @trace_run(name="ragas_eval_context_utilization", run_type="chain", tags=["ragas", "context-utilization"])
    def compute_context_utilization(
        self,
        answer: str,
        contexts: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates Context Utilization / Grader:
        Measures whether the LLM actually utilized the retrieved context blocks
        to build the final answer, penalizing systems that fetch context but ignore it.
        """
        if not contexts:
            return 0.0, {"utilized_chunks": 0, "total_chunks": 0, "utilization_per_chunk": []}

        if not answer or not answer.strip():
            return 0.0, {"utilized_chunks": 0, "total_chunks": len(contexts), "utilization_per_chunk": []}

        # Step 1: Analyze each chunk's contribution
        chunk_snippets = [c[:300] for c in contexts[:5]]
        prompt_util = (
            "You are an objective RAG Grader evaluating Context Utilization.\n"
            "Determine which of the following retrieved context chunks were actually utilized "
            "or referenced to construct the final generated answer.\n\n"
            f"Generated Answer:\n{answer[:1200]}\n\n"
            f"Retrieved Chunks:\n{json.dumps(chunk_snippets, indent=2)}\n\n"
            "Return a JSON list of objects where each item has:\n"
            "- 'chunk_index': int (0 to N-1)\n"
            "- 'utilized': bool (true if information from this chunk appears in the answer)\n"
            "- 'reason': str (brief justification)\n"
            "Output JSON only:"
        )

        raw_out = self.llm.generate(prompt_util)
        parsed = self._extract_json(raw_out)

        chunk_results = []
        if parsed and isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    chunk_results.append({
                        "chunk_index": item.get("chunk_index", 0),
                        "utilized": bool(item.get("utilized", False)),
                        "reason": str(item.get("reason", ""))
                    })

        # Fallback heuristic: keyword intersection
        if not chunk_results:
            ans_tokens = set(re.findall(r"\w+", answer.lower()))
            for idx, chunk in enumerate(chunk_snippets):
                c_tokens = set(re.findall(r"\w+", chunk.lower()))
                overlap = len(ans_tokens.intersection(c_tokens))
                is_util = overlap >= 3
                chunk_results.append({
                    "chunk_index": idx,
                    "utilized": is_util,
                    "reason": f"Token overlap count: {overlap}"
                })

        utilized_count = sum(1 for c in chunk_results if c["utilized"])
        total_chunks = len(chunk_results)
        score = round(utilized_count / max(total_chunks, 1), 3)

        details = {
            "utilized_chunks": utilized_count,
            "total_chunks": total_chunks,
            "utilization_per_chunk": chunk_results
        }
        return score, details

    # =============================================================
    # CATEGORY 2: Ground Truth-Dependent Metrics (Reference-Based)
    # =============================================================

    @trace_run(name="ragas_eval_answer_correctness", run_type="chain", tags=["ragas", "answer-correctness"])
    def compute_answer_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> Tuple[float, LLMJudgeMetrics]:
        """
        Calculates Answer Correctness (RAGAS):
        Combines factual correctness (True Positives, False Positives, False Negatives)
        with semantic similarity against the gold-standard ground truth.
        Formula: 0.75 * Factual_F1 + 0.25 * Semantic_Similarity
        """
        if not answer or not answer.strip():
            empty_judge = LLMJudgeMetrics(
                answer_correctness=0.0,
                factual_correctness_f1=0.0,
                semantic_similarity=0.0,
                true_positives=0,
                false_positives=0,
                false_negatives=1,
                details={"reason": "Empty candidate answer."}
            )
            return 0.0, empty_judge

        # Step 1: Semantic similarity calculation
        semantic_sim = compute_semantic_similarity(ground_truth, answer)

        # Step 2: Factual Claim Decomposition & Categorization (TP, FP, FN)
        prompt_f1 = (
            "You are a rigorous code verification evaluator assessing Answer Correctness.\n"
            f"Question:\n{question}\n\n"
            f"Ground Truth Reference:\n{ground_truth}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Compare the Generated Answer against the Ground Truth and identify:\n"
            "1. True Positives (TP): factual claims in the answer that correctly match the ground truth.\n"
            "2. False Positives (FP): factual claims in the answer that are contradicted by or absent from the ground truth.\n"
            "3. False Negatives (FN): important factual claims in the ground truth that the answer missed.\n\n"
            "Return JSON in the format:\n"
            "{\n"
            '  "tp": ["..."],\n'
            '  "fp": ["..."],\n'
            '  "fn": ["..."],\n'
            '  "reason": "Brief summary of factual alignment"\n'
            "}"
        )

        raw_f1 = self.llm.generate(prompt_f1)
        parsed_f1 = self._extract_json(raw_f1)

        tp_count = 0
        fp_count = 0
        fn_count = 0
        reason = ""

        if parsed_f1 and isinstance(parsed_f1, dict):
            tp_list = parsed_f1.get("tp", [])
            fp_list = parsed_f1.get("fp", [])
            fn_list = parsed_f1.get("fn", [])
            tp_count = len(tp_list) if isinstance(tp_list, list) else 0
            fp_count = len(fp_list) if isinstance(fp_list, list) else 0
            fn_count = len(fn_list) if isinstance(fn_list, list) else 0
            reason = str(parsed_f1.get("reason", ""))

        if (tp_count + fp_count + fn_count) == 0:
            # Fallback based on token overlap
            gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))
            ans_tokens = set(re.findall(r"\w+", answer.lower()))
            overlap = len(gt_tokens.intersection(ans_tokens))
            tp_count = overlap
            fp_count = max(0, len(ans_tokens) - overlap)
            fn_count = max(0, len(gt_tokens) - overlap)
            reason = "Lexical token overlap approximation"

        # F1 score: 2 * TP / (2 * TP + FP + FN)
        denom = (2 * tp_count) + fp_count + fn_count
        factual_f1 = round((2 * tp_count) / denom, 4) if denom > 0 else 0.0

        # RAGAS standard blend: 75% factual F1 + 25% semantic similarity
        overall_correctness = round((0.75 * factual_f1) + (0.25 * semantic_sim), 4)

        judge_metrics = LLMJudgeMetrics(
            answer_correctness=overall_correctness,
            factual_correctness_f1=factual_f1,
            semantic_similarity=semantic_sim,
            true_positives=tp_count,
            false_positives=fp_count,
            false_negatives=fn_count,
            details={
                "reason": reason,
                "tp_count": tp_count,
                "fp_count": fp_count,
                "fn_count": fn_count
            }
        )
        return overall_correctness, judge_metrics

    # =============================================================
    # CATEGORY 3: Retrieval-Dependent Metrics
    # =============================================================

    @trace_run(name="ragas_eval_context_recall", run_type="chain", tags=["ragas", "context-recall"])
    def compute_context_recall(
        self,
        ground_truth: str,
        contexts: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates Context Recall:
        Measures if the retrieval system successfully fetched all necessary
        information present in the ground truth to answer the question.
        Formula: (Ground Truth claims attributable to retrieved context) / (Total Ground Truth claims)
        """
        if not contexts:
            return 0.0, {"supported_gt_claims": 0, "total_gt_claims": 0, "details": []}

        if not ground_truth or not ground_truth.strip():
            return 1.0, {"supported_gt_claims": 0, "total_gt_claims": 0, "details": []}

        # Step 1: Decompose ground truth into atomic statements
        prompt_gt_claims = (
            "Decompose the following ground truth reference answer into a JSON list of discrete "
            "atomic factual statements about the codebase.\n\n"
            f"Ground Truth:\n{ground_truth}\n\n"
            "Return JSON list of strings only, for example:\n"
            '["ContextCrossChecker is defined in src/agent/context_verifier.py", "It runs verify_relevance"]'
        )

        raw_claims = self.llm.generate(prompt_gt_claims)
        parsed_claims = self._extract_json(raw_claims)

        if not parsed_claims or not isinstance(parsed_claims, list):
            parsed_claims = [
                s.strip() for s in re.split(r"[.\n]+", ground_truth)
                if len(s.strip()) > 15
            ][:4]

        parsed_claims = parsed_claims[:4]
        combined_context = "\n---\n".join([c[:750] for c in contexts[:4]])

        # Step 2: Check which ground truth claims can be inferred from retrieved context
        prompt_verify_recall = (
            "You are evaluating Retrieval Context Recall.\n"
            "For each statement from the ground truth below, determine whether it can be directly "
            "attributed to or verified using the Retrieved Code Context.\n\n"
            f"Retrieved Code Context:\n{combined_context}\n\n"
            f"Ground Truth Statements:\n{json.dumps(parsed_claims, indent=2)}\n\n"
            "Return a JSON list of objects: [{'statement': '...', 'attributed': true/false, 'evidence': '...'}]\n"
            "Output JSON only:"
        )

        raw_recall = self.llm.generate(prompt_verify_recall)
        parsed_recall = self._extract_json(raw_recall)

        recall_items = []
        if parsed_recall and isinstance(parsed_recall, list):
            for it in parsed_recall:
                if isinstance(it, dict):
                    recall_items.append({
                        "statement": it.get("statement", ""),
                        "attributed": bool(it.get("attributed", False)),
                        "evidence": it.get("evidence", "")
                    })

        if not recall_items:
            for c in parsed_claims:
                words = [w.lower() for w in re.findall(r"\w+", c) if len(w) > 3]
                matches = sum(1 for w in words if w in combined_context.lower())
                is_attr = (matches / max(len(words), 1)) >= 0.4
                recall_items.append({
                    "statement": c,
                    "attributed": is_attr,
                    "evidence": "Lexical overlap" if is_attr else "Not found in retrieved context"
                })

        supported = sum(1 for r in recall_items if r["attributed"])
        total = max(len(recall_items), 1)
        score = round(supported / total, 3)

        return score, {
            "supported_gt_claims": supported,
            "total_gt_claims": total,
            "details": recall_items
        }

    @trace_run(name="ragas_eval_context_precision", run_type="chain", tags=["ragas", "context-precision"])
    def compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Calculates Context Precision:
        Evaluates whether the most relevant information blocks were ranked highest
        in the retrieved document array, punishing systems that bury good information.
        Calculates Mean Reciprocal Rank (MRR) and rank-weighted precision.
        """
        if not contexts:
            return 0.0, "No context chunks provided."

        relevant_flags = []
        q_terms = set(re.findall(r"\w+", question.lower()))
        if ground_truth:
            q_terms.update(re.findall(r"\w+", ground_truth.lower()))

        for chunk in contexts:
            c_terms = set(re.findall(r"\w+", chunk.lower()))
            overlap = len(q_terms.intersection(c_terms))
            relevant_flags.append(1 if overlap >= 2 else 0)

        total_relevant = sum(relevant_flags)
        if total_relevant == 0:
            return 0.0, "None of the retrieved context chunks contain relevant query signals."

        # Rank-weighted precision calculation: sum(Precision@k * rel_k) / total_relevant
        precision_at_k = []
        running_rel = 0
        for k, rel in enumerate(relevant_flags, start=1):
            running_rel += rel
            precision_at_k.append(running_rel / k)

        weighted_precision = sum(p * r for p, r in zip(precision_at_k, relevant_flags)) / total_relevant
        score = round(weighted_precision, 3)

        explanation = (
            f"Context Precision: {score:.2f} | {total_relevant}/{len(contexts)} chunks relevant. "
            f"Precision@k ranks: {[round(p, 2) for p in precision_at_k]}"
        )
        return score, explanation

    @trace_run(name="ragas_eval_relevance", run_type="chain", tags=["ragas", "relevance"])
    def compute_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> Tuple[float, str]:
        """
        Calculates Answer Relevance:
        Evaluates whether the generated answer directly addresses the prompt.
        """
        if not answer or not answer.strip():
            return 0.0, "Empty answer."

        prompt_eval = (
            "You are an expert AI evaluator assessing Answer Relevance.\n"
            f"Original Question:\n{question}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Evaluate whether this answer directly, concisely, and accurately answers the question.\n"
            "Score on a scale from 0.00 to 1.00 (where 1.00 = perfect direct answer, 0.00 = completely irrelevant/hallucinated).\n"
            "Return JSON in the format:\n"
            '{"score": 0.95, "feedback": "Directly explains where CodebaseChunker is located and details AST traversal."}'
        )

        raw_output = self.llm.generate(prompt_eval)
        parsed = self._extract_json(raw_output)

        if parsed and isinstance(parsed, dict) and "score" in parsed:
            try:
                score = max(0.0, min(1.0, float(parsed["score"])))
                feedback = str(parsed.get("feedback", ""))
                return round(score, 3), feedback
            except Exception:
                pass

        q_tokens = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]
        overlap = sum(1 for w in q_tokens if w in answer.lower())
        score = min(1.0, round(overlap / max(len(q_tokens), 1), 2))
        return score, f"Keyword coverage score: {score}"

    # =============================================================
    # Unified Deep Evaluation
    # =============================================================

    @trace_run(name="ragas_full_evaluation", run_type="chain", tags=["ragas", "benchmark"])
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> RagasMetrics:
        """
        Legacy RAGAS evaluation runner:
        Faithfulness (0.45) + Answer Relevance (0.35) + Context Precision (0.20)
        """
        faithfulness_score, claims = self.compute_faithfulness(question, answer, contexts)
        relevance_score, rel_feedback = self.compute_answer_relevance(question, answer)
        precision_score, prec_feedback = self.compute_context_precision(question, contexts)

        composite = round(
            (faithfulness_score * 0.45) + (relevance_score * 0.35) + (precision_score * 0.20),
            3
        )

        supported_claims = sum(1 for c in claims if c.supported)
        relevant_chunks = int(precision_score * len(contexts))

        summary_feedback = (
            f"Faithfulness: {faithfulness_score:.2f} ({supported_claims}/{len(claims)} claims verified) | "
            f"Relevance: {relevance_score:.2f} | "
            f"Context Precision: {precision_score:.2f} | "
            f"RAG Quality: {composite:.2f}/1.00. Notes: {rel_feedback}"
        )

        metrics = RagasMetrics(
            faithfulness=faithfulness_score,
            answer_relevance=relevance_score,
            context_precision=precision_score,
            composite_score=composite,
            claims_evaluated=len(claims),
            claims_supported=supported_claims,
            claims_detail=[asdict(c) for c in claims],
            context_chunks_evaluated=len(contexts),
            context_chunks_relevant=relevant_chunks,
            feedback=summary_feedback
        )

        return metrics

    @trace_run(name="deep_rag_evaluation", run_type="chain", tags=["evaluation", "deep-metrics"])
    def evaluate_deep(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> DeepEvaluationReport:
        """
        Executes comprehensive multi-category evaluation covering all dependent metrics:
        1. Context-Dependent: Faithfulness & Context Utilization / Grader
        2. Ground Truth-Dependent: LLM-as-a-Judge (Answer Correctness & Semantic Sim) + Lexical (BLEU, ROUGE, METEOR)
        3. Retrieval-Dependent: Context Recall & Rank-Weighted Context Precision (with MRR)
        """
        # 1. Context-Dependent Metrics
        faithfulness, claims = self.compute_faithfulness(question, answer, contexts)
        utilization, util_details = self.compute_context_utilization(answer, contexts)
        supported_claims = sum(1 for c in claims if c.supported)

        context_dependent = ContextDependentMetrics(
            faithfulness=faithfulness,
            context_utilization=utilization,
            claims_evaluated=len(claims),
            claims_supported=supported_claims,
            utilized_chunks=util_details["utilized_chunks"],
            total_chunks=len(contexts),
            details={"claims": [asdict(c) for c in claims], "utilization": util_details}
        )

        # 2. Ground Truth-Dependent Metrics (if ground_truth provided)
        gt_dependent = None
        if ground_truth and ground_truth.strip():
            _, judge_metrics = self.compute_answer_correctness(question, answer, ground_truth)
            lexical_metrics = compute_all_lexical_metrics(ground_truth, answer)

            gt_dependent = GroundTruthDependentMetrics(
                llm_judge=judge_metrics,
                lexical=lexical_metrics
            )

        # 3. Retrieval-Dependent Metrics
        recall_score, recall_details = (
            self.compute_context_recall(ground_truth, contexts)
            if ground_truth
            else (1.0, {"supported_gt_claims": 0, "total_gt_claims": 0, "details": []})
        )
        prec_score, prec_feedback = self.compute_context_precision(question, contexts, ground_truth=ground_truth)

        # Compute MRR (Mean Reciprocal Rank of first relevant chunk)
        mrr = 0.0
        q_terms = set(re.findall(r"\w+", question.lower()))
        for rank, chunk in enumerate(contexts, start=1):
            c_terms = set(re.findall(r"\w+", chunk.lower()))
            if len(q_terms.intersection(c_terms)) >= 2:
                mrr = round(1.0 / rank, 3)
                break

        retrieval_dependent = RetrievalDependentMetrics(
            context_recall=recall_score,
            context_precision=prec_score,
            mean_reciprocal_rank=mrr,
            total_chunks=len(contexts),
            relevant_chunks=recall_details.get("supported_gt_claims", 0),
            details={"recall": recall_details, "precision_feedback": prec_feedback}
        )

        # Composite multi-dimensional score
        if gt_dependent:
            composite = round(
                (faithfulness * 0.25)
                + (utilization * 0.15)
                + (gt_dependent.llm_judge.answer_correctness * 0.30)
                + (recall_score * 0.15)
                + (prec_score * 0.15),
                3
            )
        else:
            composite = round(
                (faithfulness * 0.40) + (utilization * 0.30) + (prec_score * 0.30),
                3
            )

        summary_lines = [
            f"Context-Dependent: Faithfulness={faithfulness:.2f}, Utilization={utilization:.2f}",
            f"Retrieval-Dependent: Recall={recall_score:.2f}, Precision={prec_score:.2f}, MRR={mrr:.2f}"
        ]
        if gt_dependent:
            summary_lines.append(
                f"Ground-Truth Dependent: Correctness={gt_dependent.llm_judge.answer_correctness:.2f}, "
                f"SemanticSim={gt_dependent.llm_judge.semantic_similarity:.2f}, "
                f"ROUGE-L={gt_dependent.lexical['rouge_l']['f1']:.2f}, BLEU={gt_dependent.lexical['bleu']:.2f}, "
                f"METEOR={gt_dependent.lexical['meteor']:.2f}"
            )
        summary_lines.append(f"Overall Deep Quality Score: {composite:.2f}/1.00")

        return DeepEvaluationReport(
            context_dependent=context_dependent,
            ground_truth_dependent=gt_dependent,
            retrieval_dependent=retrieval_dependent,
            composite_score=composite,
            summary=" | ".join(summary_lines)
        )
