"""RAGEvaluator: run a retriever on a dataset and compute all metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag_eval.datasets.loader import DatasetLoader, QAPair
from graph_rag_eval.metrics.efficiency import EfficiencyMetrics
from graph_rag_eval.metrics.graph_specific import avg_hop_distance, entity_coverage, relation_coverage
from graph_rag_eval.metrics.retrieval import ndcg, precision_at_k, recall_at_k, mrr
from graph_rag_eval.retrievers.base import BaseRetriever, Document


@dataclass
class QueryResult:
    """Result for a single query."""

    query: str
    answer: str
    retrieved: List[Document]
    relevant_ids: List[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    entity_coverage: float
    relation_coverage: float
    avg_hop_distance: float
    latency_ms: float


@dataclass
class EvalReport:
    """
    Aggregated evaluation report for a retriever on a dataset.

    All metric values are mean across queries.
    """

    retriever_name: str
    dataset_name: str
    k: int
    num_queries: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_ndcg: float
    mean_entity_coverage: float
    mean_relation_coverage: float
    mean_avg_hop_distance: float
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    query_results: List[QueryResult] = field(default_factory=list)
    multihop_entity_coverage: float = 0.0
    singlehop_entity_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retriever": self.retriever_name,
            "dataset": self.dataset_name,
            "k": self.k,
            "num_queries": self.num_queries,
            "recall@k": round(self.mean_recall_at_k, 4),
            f"recall@{self.k}": round(self.mean_recall_at_k, 4),
            "precision@k": round(self.mean_precision_at_k, 4),
            "mrr": round(self.mean_mrr, 4),
            "ndcg": round(self.mean_ndcg, 4),
            "entity_coverage": round(self.mean_entity_coverage, 4),
            "relation_coverage": round(self.mean_relation_coverage, 4),
            "avg_hop_distance": round(self.mean_avg_hop_distance, 4),
            "multihop_entity_coverage": round(self.multihop_entity_coverage, 4),
            "singlehop_entity_coverage": round(self.singlehop_entity_coverage, 4),
            "efficiency": self.efficiency.to_dict(),
        }

    def summary(self) -> str:
        lines = [
            f"=== {self.retriever_name} on {self.dataset_name} (k={self.k}) ===",
            f"  Queries:           {self.num_queries}",
            f"  Recall@{self.k}:          {self.mean_recall_at_k:.3f}",
            f"  Precision@{self.k}:       {self.mean_precision_at_k:.3f}",
            f"  MRR:               {self.mean_mrr:.3f}",
            f"  NDCG:              {self.mean_ndcg:.3f}",
            f"  Entity Coverage:   {self.mean_entity_coverage:.3f}",
            f"  Relation Coverage: {self.mean_relation_coverage:.3f}",
            f"  Avg Hop Distance:  {self.mean_avg_hop_distance:.3f}",
            f"  Multi-hop Ent Cov: {self.multihop_entity_coverage:.3f}",
            f"  Single-hop Ent Cov:{self.singlehop_entity_coverage:.3f}",
            f"  Latency (mean):    {self.efficiency.mean_latency_ms:.2f}ms",
        ]
        return "\n".join(lines)


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class RAGEvaluator:
    """
    Evaluate a retriever on a dataset and compute comprehensive metrics.

    Usage::

        from graph_rag_eval import RAGEvaluator
        from graph_rag_eval.retrievers import GraphRetriever
        from graph_rag_eval.datasets import SyntheticQADataset

        dataset = SyntheticQADataset()
        retriever = GraphRetriever(dataset.knowledge_graph)
        retriever.build_index([])

        evaluator = RAGEvaluator(retriever, "GraphRetriever")
        report = evaluator.evaluate(dataset, k=5)
        print(report.summary())
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        retriever_name: str = "retriever",
    ) -> None:
        self._retriever = retriever
        self._name = retriever_name

    def evaluate(
        self,
        dataset: DatasetLoader,
        k: int = 5,
        dataset_name: str = "dataset",
        verbose: bool = False,
    ) -> EvalReport:
        """
        Run the retriever on every QA pair and aggregate metrics.

        Args:
            dataset: Dataset to evaluate on.
            k: Number of documents to retrieve per query.
            dataset_name: Label for the dataset in the report.
            verbose: Print per-query results.

        Returns:
            EvalReport with all metrics.
        """
        pairs = dataset.load()
        efficiency = EfficiencyMetrics(index_size=self._retriever.index_size)

        recalls, precisions, mrrs, ndcgs = [], [], [], []
        entity_covs, relation_covs, hop_dists = [], [], []
        query_results = []
        multihop_entity_covs, singlehop_entity_covs = [], []

        for pair in pairs:
            docs = self._retriever.retrieve(pair.question, k=k)
            efficiency.record(self._retriever)

            retrieved_ids = [d.id for d in docs]

            r_k = recall_at_k(pair.relevant_ids, retrieved_ids, k)
            p_k = precision_at_k(pair.relevant_ids, retrieved_ids, k)
            mrr_score = mrr(pair.relevant_ids, retrieved_ids)
            ndcg_score = ndcg(pair.relevant_ids, retrieved_ids, k)
            ent_cov = entity_coverage(docs, pair.answer_entities)
            rel_cov = relation_coverage(docs, pair.answer_relations)
            hop_dist = avg_hop_distance(docs, pair.answer_entities)

            recalls.append(r_k)
            precisions.append(p_k)
            mrrs.append(mrr_score)
            ndcgs.append(ndcg_score)
            entity_covs.append(ent_cov)
            relation_covs.append(rel_cov)
            if hop_dist >= 0:
                hop_dists.append(hop_dist)

            if pair.is_multihop():
                multihop_entity_covs.append(ent_cov)
            else:
                singlehop_entity_covs.append(ent_cov)

            qr = QueryResult(
                query=pair.question,
                answer=pair.answer,
                retrieved=docs,
                relevant_ids=pair.relevant_ids,
                recall_at_k=r_k,
                precision_at_k=p_k,
                mrr=mrr_score,
                ndcg=ndcg_score,
                entity_coverage=ent_cov,
                relation_coverage=rel_cov,
                avg_hop_distance=hop_dist,
                latency_ms=self._retriever.last_latency_ms,
            )
            query_results.append(qr)

            if verbose:
                print(f"Q: {pair.question[:70]}")
                print(f"   Recall@{k}={r_k:.2f}  EntCov={ent_cov:.2f}  Hop={hop_dist:.1f}")

        return EvalReport(
            retriever_name=self._name,
            dataset_name=dataset_name,
            k=k,
            num_queries=len(pairs),
            mean_recall_at_k=_safe_mean(recalls),
            mean_precision_at_k=_safe_mean(precisions),
            mean_mrr=_safe_mean(mrrs),
            mean_ndcg=_safe_mean(ndcgs),
            mean_entity_coverage=_safe_mean(entity_covs),
            mean_relation_coverage=_safe_mean(relation_covs),
            mean_avg_hop_distance=_safe_mean(hop_dists),
            efficiency=efficiency,
            query_results=query_results,
            multihop_entity_coverage=_safe_mean(multihop_entity_covs),
            singlehop_entity_coverage=_safe_mean(singlehop_entity_covs),
        )
