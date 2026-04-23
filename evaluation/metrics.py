"""
Evaluation metrics for DS Mentor Pro.

Design goals:
- lightweight (stdlib-first) with optional third-party acceleration
- explicit, unit-testable metric functions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Ranking / retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    """Precision@k for a single ranked list.

    Args:
        relevances: list where 1 = relevant, 0 = not relevant, in rank order.
        k: cutoff.
    """
    k = max(1, int(k))
    topk = list(relevances)[:k]
    if not topk:
        return 0.0
    return float(sum(1 for r in topk if r)) / float(len(topk))


def recall_at_k(relevances: Sequence[int], k: int, n_relevant: int) -> float:
    """Recall@k for a single ranked list given total # relevant items."""
    k = max(1, int(k))
    n_relevant = int(n_relevant)
    if n_relevant <= 0:
        return 0.0
    topk = list(relevances)[:k]
    return float(sum(1 for r in topk if r)) / float(n_relevant)


def reciprocal_rank(relevances: Sequence[int]) -> float:
    """Reciprocal rank (RR): 1/rank of first relevant result, else 0."""
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / float(idx)
    return 0.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain at k using binary relevances."""
    import math

    k = max(1, int(k))
    dcg = 0.0
    for i, rel in enumerate(list(relevances)[:k], start=1):
        if rel:
            dcg += 1.0 / math.log2(i + 1.0)
    return float(dcg)


def ndcg_at_k(relevances: Sequence[int], k: int, n_relevant: int) -> float:
    """Normalized DCG at k for binary relevances and known n_relevant."""
    k = max(1, int(k))
    n_relevant = max(0, int(n_relevant))
    dcg = dcg_at_k(relevances, k)
    ideal_rels = [1] * min(k, n_relevant) + [0] * max(0, k - min(k, n_relevant))
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# Classification + detection metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


def prf1_from_counts(tp: int, fp: int, fn: int) -> PRF1:
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PRF1(precision=float(precision), recall=float(recall), f1=float(f1))


def confusion_matrix_counts(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None
) -> Tuple[List[int], List[List[int]]]:
    """Return (labels, matrix) where matrix[i][j] = count(true=labels[i], pred=labels[j])."""
    if labels is None:
        labels = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    labels = [int(x) for x in labels]
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        i = idx[int(yt)]
        j = idx[int(yp)]
        m[i][j] += 1
    return labels, m


# ---------------------------------------------------------------------------
# Simple text overlap metrics (response quality)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> List[str]:
    import re

    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def bleu1(reference: str, hypothesis: str) -> float:
    """Very small BLEU-1 (unigram precision with brevity penalty)."""
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    if not hyp:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    hyp_counts: Dict[str, int] = {}
    for t in hyp:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1
    clipped = 0
    for t, c in hyp_counts.items():
        clipped += min(c, ref_counts.get(t, 0))
    prec = clipped / float(sum(hyp_counts.values()))
    bp = min(1.0, len(hyp) / float(max(1, len(ref))))
    return float(bp * prec)


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 based on token LCS (lightweight)."""
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    if not ref or not hyp:
        return 0.0

    # LCS length DP (O(n*m) but small for our eval set)
    n, m = len(ref), len(hyp)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[m]
    prec = lcs / float(m)
    rec = lcs / float(n)
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# ---------------------------------------------------------------------------
# Lightweight "CodeBLEU-like" score (NOT the official CodeBLEU paper metric)
# ---------------------------------------------------------------------------


def extract_code_keywords(code: str) -> set[str]:
    """Extract Python-ish keyword tokens for a coarse overlap signal."""
    import re

    patterns = [
        r"\b(def|class|import|from|return|yield|lambda)\b",
        r"\b\w+\(\)",  # function calls (very rough)
        r"\b[A-Z][a-zA-Z]+\b",  # CamelCase
        r"\.\w+\(",  # method calls
    ]
    tokens: set[str] = set()
    for p in patterns:
        tokens.update(re.findall(p, code or ""))
    return tokens


def codebleu_simple(reference: str, hypothesis: str) -> float:
    """
    Simplified Code-style score (same spirit as `evaluation/benchmark.compute_codebleu`):
      0.25 * bleu1 + 0.25 * keyword_match + 0.25 * syntax_valid + 0.25 * ast_node_overlap

    Notes:
      - This is a lightweight approximation for reporting.
      - Official CodeBLEU uses grammar/dataflow signals; we do not claim full parity.
    """
    import ast

    bleu = float(bleu1(reference, hypothesis))

    ref_kws = extract_code_keywords(reference)
    hyp_kws = extract_code_keywords(hypothesis)
    if ref_kws:
        kw_match = len(ref_kws & hyp_kws) / float(len(ref_kws))
    else:
        kw_match = 0.5

    try:
        ast.parse(hypothesis or "")
        syntax_score = 1.0
    except SyntaxError:
        syntax_score = 0.0

    def _ast_nodes(code: str) -> set[str]:
        try:
            tree = ast.parse(code or "")
            return {type(node).__name__ for node in ast.walk(tree)}
        except Exception:
            return set()

    ref_nodes = _ast_nodes(reference)
    hyp_nodes = _ast_nodes(hypothesis)
    if ref_nodes:
        node_overlap = len(ref_nodes & hyp_nodes) / float(len(ref_nodes))
    else:
        node_overlap = 0.5

    return float(
        round(0.25 * bleu + 0.25 * kw_match + 0.25 * syntax_score + 0.25 * node_overlap, 4)
    )

