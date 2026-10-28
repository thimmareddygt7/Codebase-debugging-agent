import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Any


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return [w.lower() for w in re.findall(r"\w+", text) if w]


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generates n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# 1. BLEU Score (Bilingual Evaluation Understudy)
# ---------------------------------------------------------------------------
def compute_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4
) -> Dict[str, float]:
    """
    Computes BLEU metric with modified n-gram precision and brevity penalty.
    Returns:
        bleu: Cumulative geometric mean BLEU score (0.0 to 1.0)
        bleu_1, bleu_2, bleu_3, bleu_4: Individual n-gram precision scores
        brevity_penalty: Length penalty factor
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)} | {"bleu": 0.0, "brevity_penalty": 0.0}

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    # Brevity Penalty
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - (float(ref_len) / float(hyp_len)))

    p_ns = []
    scores = {}

    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_get_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_get_ngrams(hyp_tokens, n))

        total_hyp_ngrams = sum(hyp_ngrams.values())
        if total_hyp_ngrams == 0:
            p_n = 0.0
        else:
            # Clipped counts
            overlap = 0
            for ng, count in hyp_ngrams.items():
                overlap += min(count, ref_ngrams.get(ng, 0))
            p_n = float(overlap) / float(total_hyp_ngrams)

        scores[f"bleu_{n}"] = round(p_n, 4)
        p_ns.append(p_n)

    # Cumulative BLEU (smoothed geometric mean)
    valid_p = [p for p in p_ns if p > 0]
    if len(valid_p) == len(p_ns) and all(p > 0 for p in p_ns):
        log_avg = sum(math.log(p) for p in p_ns) / float(len(p_ns))
        bleu = bp * math.exp(log_avg)
    elif p_ns[0] > 0:
        # Fallback smoothing for short code texts
        weights = [0.5, 0.3, 0.15, 0.05][:len(p_ns)]
        smooth_score = sum(w * p for w, p in zip(weights, p_ns))
        bleu = bp * smooth_score
    else:
        bleu = 0.0

    scores["bleu"] = round(bleu, 4)
    scores["brevity_penalty"] = round(bp, 4)
    return scores


# ---------------------------------------------------------------------------
# 2. ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)
# ---------------------------------------------------------------------------
def _lcs_length(x: List[str], y: List[str]) -> int:
    """Computes the length of Longest Common Subsequence via dynamic programming."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge(
    reference: str,
    hypothesis: str
) -> Dict[str, Dict[str, float]]:
    """
    Computes ROUGE metrics:
        - ROUGE-1 (Unigram overlap)
        - ROUGE-2 (Bigram overlap)
        - ROUGE-L (Longest Common Subsequence)
    Each with precision (p), recall (r), and F1 measure (f1).
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    results = {}

    for name, n in [("rouge_1", 1), ("rouge_2", 2)]:
        ref_ngrams = Counter(_get_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_get_ngrams(hyp_tokens, n))

        total_ref = sum(ref_ngrams.values())
        total_hyp = sum(hyp_ngrams.values())

        if total_ref == 0 or total_hyp == 0:
            results[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        p = overlap / float(total_hyp)
        r = overlap / float(total_ref)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        results[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    # ROUGE-L (Longest Common Subsequence)
    if not ref_tokens or not hyp_tokens:
        results["rouge_l"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        lcs = _lcs_length(ref_tokens, hyp_tokens)
        p = lcs / float(len(hyp_tokens))
        r = lcs / float(len(ref_tokens))
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        results["rouge_l"] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    return results


# ---------------------------------------------------------------------------
# 3. METEOR Metric (Harmonic mean with fragmentation penalty)
# ---------------------------------------------------------------------------
def compute_meteor(
    reference: str,
    hypothesis: str,
    alpha: float = 0.9,
    gamma: float = 0.5,
    beta: float = 3.0
) -> float:
    """
    Computes METEOR metric incorporating unigram precision, recall,
    and a chunk fragmentation penalty.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # Unigram matching
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    matches = 0
    for tok, count in hyp_counts.items():
        matches += min(count, ref_counts.get(tok, 0))

    if matches == 0:
        return 0.0

    p = float(matches) / float(len(hyp_tokens))
    r = float(matches) / float(len(ref_tokens))

    # Harmonic mean weighted towards recall (alpha = 0.9)
    f_mean = (p * r) / ((1 - alpha) * r + alpha * p)

    # Calculate fragmentation penalty based on matching contiguous chunks
    chunks = 0
    in_chunk = False
    ref_set = set(ref_tokens)

    for tok in hyp_tokens:
        if tok in ref_set:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False

    chunks = max(1, chunks)
    frag_ratio = float(chunks) / float(matches)
    penalty = gamma * (frag_ratio ** beta)

    score = f_mean * (1.0 - penalty)
    return round(max(0.0, min(1.0, score)), 4)


def compute_all_lexical_metrics(
    reference: str,
    hypothesis: str
) -> Dict[str, Any]:
    """
    Computes all classic lexical and statistical metrics:
    - BLEU (1, 2, 3, 4, cumulative, brevity penalty)
    - ROUGE (rouge_1, rouge_2, rouge_l precision, recall, f1)
    - METEOR (chunk-penalized harmonic recall-weighted score)
    """
    bleu_scores = compute_bleu(reference, hypothesis)
    rouge_scores = compute_rouge(reference, hypothesis)
    meteor_score = compute_meteor(reference, hypothesis)

    return {
        "bleu": bleu_scores["bleu"],
        "bleu_1": bleu_scores["bleu_1"],
        "bleu_2": bleu_scores["bleu_2"],
        "bleu_3": bleu_scores["bleu_3"],
        "bleu_4": bleu_scores["bleu_4"],
        "brevity_penalty": bleu_scores["brevity_penalty"],
        "rouge_1": rouge_scores["rouge_1"],
        "rouge_2": rouge_scores["rouge_2"],
        "rouge_l": rouge_scores["rouge_l"],
        "meteor": meteor_score,
    }

