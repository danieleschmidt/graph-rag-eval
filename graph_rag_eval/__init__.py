"""
graph-rag-eval: Evaluation Framework for Graph-Augmented RAG

Compare graph-aware retrieval over knowledge graphs vs flat TF-IDF retrieval
with standardized metrics: Recall@K, NDCG, entity coverage, relation coverage.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from graph_rag_eval.evaluator import RAGEvaluator
from graph_rag_eval.comparison import ComparisonReport

__all__ = ["RAGEvaluator", "ComparisonReport"]
