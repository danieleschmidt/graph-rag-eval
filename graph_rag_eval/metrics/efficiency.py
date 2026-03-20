"""Efficiency metrics: latency, index size, nodes visited."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

from graph_rag_eval.retrievers.base import BaseRetriever


@dataclass
class EfficiencyMetrics:
    """
    Measured efficiency metrics for a retriever evaluation run.

    Attributes:
        latencies_ms: Per-query latency in milliseconds.
        nodes_visited: Nodes visited per query (graph retrievers only).
        index_size: Number of documents/nodes in the index.
    """

    latencies_ms: List[float] = field(default_factory=list)
    nodes_visited: List[int] = field(default_factory=list)
    index_size: int = 0

    def record(self, retriever: BaseRetriever) -> None:
        """Record latency and nodes_visited from a retriever after retrieve()."""
        self.latencies_ms.append(retriever.last_latency_ms)
        self.nodes_visited.append(retriever.last_nodes_visited)
        self.index_size = retriever.index_size

    @property
    def mean_latency_ms(self) -> float:
        """Mean query latency in milliseconds."""
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile query latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        return sorted_lat[idx]

    @property
    def mean_nodes_visited(self) -> float:
        """Mean nodes visited per query."""
        return statistics.mean(self.nodes_visited) if self.nodes_visited else 0.0

    @property
    def total_queries(self) -> int:
        """Total number of queries recorded."""
        return len(self.latencies_ms)

    def to_dict(self) -> dict:
        return {
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "index_size": self.index_size,
            "mean_nodes_visited": round(self.mean_nodes_visited, 1),
            "total_queries": len(self.latencies_ms),
        }


def benchmark_retriever(
    retriever: BaseRetriever,
    queries: List[str],
    k: int = 5,
    warmup: int = 1,
) -> EfficiencyMetrics:
    """
    Benchmark a retriever on a list of queries.

    Args:
        retriever: Built retriever to benchmark.
        queries: List of query strings.
        k: Number of documents to retrieve.
        warmup: Number of warmup queries to run before recording.

    Returns:
        EfficiencyMetrics populated with per-query measurements.
    """
    metrics = EfficiencyMetrics(index_size=retriever.index_size)

    # Warmup
    for q in queries[:warmup]:
        retriever.retrieve(q, k=k)

    # Measurement
    for q in queries:
        retriever.retrieve(q, k=k)
        metrics.record(retriever)

    return metrics
