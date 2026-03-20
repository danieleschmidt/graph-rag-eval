"""Answer quality metrics: exact match, F1 overlap, answer coverage."""

from __future__ import annotations

import re
from typing import List, Set


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Binary exact match after normalization.

    Returns:
        1.0 if normalized strings match, 0.0 otherwise.
    """
    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0


def f1_overlap(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 overlap between prediction and ground truth.

    Useful for QA tasks where partial credit is warranted.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = _tokens(prediction)
    gt_tokens = _tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    common = pred_set & gt_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_coverage(
    retrieved_texts: List[str], ground_truth_answer: str
) -> float:
    """
    Fraction of ground-truth answer tokens covered by the retrieved texts.

    Measures whether the retriever surfaced enough content to answer the question.

    Args:
        retrieved_texts: List of retrieved document contents.
        ground_truth_answer: The correct answer text.

    Returns:
        Coverage score in [0, 1].
    """
    gt_tokens = set(_tokens(ground_truth_answer))
    if not gt_tokens:
        return 0.0

    combined = " ".join(retrieved_texts)
    retrieved_tokens = set(_tokens(combined))
    covered = gt_tokens & retrieved_tokens
    return len(covered) / len(gt_tokens)
