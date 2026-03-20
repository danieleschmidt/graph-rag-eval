"""ComparisonReport: compare flat vs graph retriever side by side."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graph_rag_eval.evaluator import EvalReport


METRIC_LABELS = {
    "mean_recall_at_k": "Recall@K",
    "mean_precision_at_k": "Precision@K",
    "mean_mrr": "MRR",
    "mean_ndcg": "NDCG",
    "mean_entity_coverage": "Entity Coverage",
    "mean_relation_coverage": "Relation Coverage",
    "multihop_entity_coverage": "Multi-hop Entity Coverage",
    "singlehop_entity_coverage": "Single-hop Entity Coverage",
}


@dataclass
class ComparisonReport:
    """
    Side-by-side comparison of two EvalReports (typically flat vs graph).

    Usage::

        comparison = ComparisonReport(flat_report, graph_report)
        print(comparison.summary())
        comparison.save("report.json")
    """

    baseline: EvalReport
    candidate: EvalReport
    label_baseline: str = "Baseline"
    label_candidate: str = "Candidate"

    def __post_init__(self) -> None:
        if not self.label_baseline:
            self.label_baseline = self.baseline.retriever_name
        if not self.label_candidate:
            self.label_candidate = self.candidate.retriever_name

    def delta(self, metric: str) -> float:
        """Return candidate - baseline for a named metric."""
        base_val = getattr(self.baseline, metric, 0.0)
        cand_val = getattr(self.candidate, metric, 0.0)
        return cand_val - base_val

    def relative_improvement(self, metric: str) -> float:
        """Return % relative improvement of candidate over baseline."""
        base_val = getattr(self.baseline, metric, 0.0)
        cand_val = getattr(self.candidate, metric, 0.0)
        if base_val == 0:
            return float("inf") if cand_val > 0 else 0.0
        return (cand_val - base_val) / base_val * 100

    def summary(self) -> str:
        """Formatted text comparison table."""
        col_w = 26
        lines = [
            "=" * 80,
            f"  Graph-RAG Evaluation Comparison Report",
            f"  Dataset: {self.baseline.dataset_name}  |  k={self.baseline.k}",
            "=" * 80,
            f"{'Metric':<{col_w}} {self.label_baseline:>12} {self.label_candidate:>12} {'Delta':>10} {'Δ%':>8}",
            "-" * 80,
        ]

        for attr, label in METRIC_LABELS.items():
            base_val = getattr(self.baseline, attr, 0.0)
            cand_val = getattr(self.candidate, attr, 0.0)
            delta = cand_val - base_val
            rel = self.relative_improvement(attr)
            sign = "+" if delta >= 0 else ""
            rel_sign = "+" if rel >= 0 else ""
            lines.append(
                f"{label:<{col_w}} {base_val:>12.3f} {cand_val:>12.3f} "
                f"{sign}{delta:>9.3f} {rel_sign}{rel:>6.1f}%"
            )

        lines.append("-" * 80)

        # Efficiency comparison
        base_eff = self.baseline.efficiency
        cand_eff = self.candidate.efficiency
        lines.append(
            f"{'Latency (mean ms)':<{col_w}} {base_eff.mean_latency_ms:>12.2f} "
            f"{cand_eff.mean_latency_ms:>12.2f}"
        )
        lines.append(
            f"{'Nodes Visited (mean)':<{col_w}} {base_eff.mean_nodes_visited:>12.1f} "
            f"{cand_eff.mean_nodes_visited:>12.1f}"
        )
        lines.append("=" * 80)

        # Verdict
        multihop_delta = self.delta("multihop_entity_coverage")
        if multihop_delta > 0:
            lines.append(
                f"\n✓ {self.label_candidate} outperforms {self.label_baseline} on "
                f"multi-hop queries by +{multihop_delta:.1%} entity coverage."
            )
        elif multihop_delta < 0:
            lines.append(
                f"\n✗ {self.label_baseline} outperforms {self.label_candidate} on "
                f"multi-hop queries by {abs(multihop_delta):.1%} entity coverage."
            )
        else:
            lines.append(f"\n~ {self.label_baseline} and {self.label_candidate} are equivalent.")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize comparison to a dict (suitable for JSON output)."""
        result: Dict[str, Any] = {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "comparison": {},
        }
        for attr, label in METRIC_LABELS.items():
            result["comparison"][label] = {
                "baseline": round(getattr(self.baseline, attr, 0.0), 4),
                "candidate": round(getattr(self.candidate, attr, 0.0), 4),
                "delta": round(self.delta(attr), 4),
                "relative_pct": round(self.relative_improvement(attr), 2),
            }
        return result

    def save(self, path: str) -> None:
        """Write comparison report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Report saved to {path}")

    def winner(self, metric: str = "mean_entity_coverage") -> str:
        """Return name of the better retriever for the given metric."""
        delta = self.delta(metric)
        if delta > 0:
            return self.label_candidate
        elif delta < 0:
            return self.label_baseline
        else:
            return "tie"
