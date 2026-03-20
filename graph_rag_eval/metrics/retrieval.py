"""Standard IR retrieval metrics: Recall@K, Precision@K, MRR, NDCG."""

from __future__ import annotations

import math
from typing import List, Set


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """
    Fraction of relevant documents that appear in the top-k retrieved.

    Args:
        relevant: Ground-truth relevant document IDs.
        retrieved: Retrieved document IDs (ranked, best first).
        k: Cutoff rank.

    Returns:
        Recall@K in [0, 1].

    Example::

        recall_at_k(["a", "b", "c"], ["a", "x", "b"], k=3)
        # → 0.667  (2 of 3 relevant found in top 3)
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """
    Fraction of top-k retrieved documents that are relevant.

    Args:
        relevant: Ground-truth relevant document IDs.
        retrieved: Retrieved document IDs (ranked, best first).
        k: Cutoff rank.

    Returns:
        Precision@K in [0, 1].
    """
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / k


def mrr(relevant: List[str], retrieved: List[str]) -> float:
    """
    Mean Reciprocal Rank for a single query.

    Args:
        relevant: Ground-truth relevant document IDs.
        retrieved: Retrieved document IDs (ranked, best first).

    Returns:
        MRR score in [0, 1]. Returns 0 if no relevant doc found.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg(relevant: List[str], retrieved: List[str], k: int | None = None) -> float:
    """
    Normalized Discounted Cumulative Gain.

    Binary relevance: relevant docs have gain=1, others gain=0.

    Args:
        relevant: Ground-truth relevant document IDs.
        retrieved: Retrieved document IDs (ranked, best first).
        k: Cutoff rank. If None, uses len(retrieved).

    Returns:
        NDCG score in [0, 1].
    """
    if not relevant or not retrieved:
        return 0.0

    cutoff = k if k is not None else len(retrieved)
    relevant_set = set(relevant)

    def dcg(ranking: List[str]) -> float:
        total = 0.0
        for i, doc_id in enumerate(ranking[:cutoff], start=1):
            if doc_id in relevant_set:
                total += 1.0 / math.log2(i + 1)
        return total

    actual_dcg = dcg(retrieved)
    # Ideal DCG: place all relevant docs at the top
    ideal_ranking = [r for r in relevant if r in relevant_set][:cutoff]
    ideal_dcg = dcg(ideal_ranking)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(relevant: List[str], retrieved: List[str]) -> float:
    """
    Average Precision for a single query.

    Args:
        relevant: Ground-truth relevant document IDs.
        retrieved: Retrieved document IDs (ranked, best first).

    Returns:
        AP score in [0, 1].
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    total_precision = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            total_precision += hits / rank
    return total_precision / len(relevant)
