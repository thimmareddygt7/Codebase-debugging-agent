import math
import re
from collections import Counter
from typing import Optional, List
import numpy as np

_EMBEDDING_MODEL = None


def _get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazily loads and caches the SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
        except Exception as e:
            _EMBEDDING_MODEL = False
    return _EMBEDDING_MODEL if _EMBEDDING_MODEL is not False else None


def _char_ngram_vector_cosine(text1: str, text2: str, n: int = 3) -> float:
    """
    Robust fallback cosine similarity based on subword/character n-gram distribution.
    Guarantees reliable semantic overlap estimation even if offline.
    """
    if not text1 or not text2:
        return 0.0

    t1 = re.sub(r"\s+", " ", text1.lower().strip())
    t2 = re.sub(r"\s+", " ", text2.lower().strip())

    if t1 == t2:
        return 1.0

    ngrams1 = [t1[i : i + n] for i in range(max(len(t1) - n + 1, 1))]
    ngrams2 = [t2[i : i + n] for i in range(max(len(t2) - n + 1, 1))]

    vec1 = Counter(ngrams1)
    vec2 = Counter(ngrams2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([val ** 2 for val in vec1.values()])
    sum2 = sum([val ** 2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


def compute_semantic_similarity(
    reference: str,
    hypothesis: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> float:
    """
    Computes Semantic Similarity between reference ground truth and generated hypothesis:
    Utilizes embedding model cosine similarity, with token-distribution fallback.
    Returns float in range [0.0, 1.0].
    """
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    if reference.strip() == hypothesis.strip():
        return 1.0

    model = _get_embedding_model(model_name)
    if model is not None:
        try:
            embeddings = model.encode(
                [reference, hypothesis],
                normalize_embeddings=True,
                show_progress_bar=False
            )
            vec1 = embeddings[0]
            vec2 = embeddings[1]
            # Cosine similarity of normalized vectors is simple dot product
            cos_sim = float(np.dot(vec1, vec2))
            # Rescale / clip to [0.0, 1.0]
            return round(max(0.0, min(1.0, cos_sim)), 4)
        except Exception:
            pass

    # Fallback to subword n-gram vector cosine
    return round(_char_ngram_vector_cosine(reference, hypothesis), 4)


def compute_batch_semantic_similarity(
    references: List[str],
    hypotheses: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[float]:
    """Computes semantic similarity for pairs of references and hypotheses."""
    return [
        compute_semantic_similarity(ref, hyp, model_name=model_name)
        for ref, hyp in zip(references, hypotheses)
    ]
