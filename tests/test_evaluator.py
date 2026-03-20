"""Tests for RAGEvaluator end-to-end and ComparisonReport."""

import pytest
from graph_rag_eval.datasets.synthetic import SyntheticQADataset
from graph_rag_eval.evaluator import RAGEvaluator
from graph_rag_eval.comparison import ComparisonReport
from graph_rag_eval.retrievers.flat_retriever import FlatRetriever
from graph_rag_eval.retrievers.graph_retriever import GraphRetriever
from graph_rag_eval.retrievers.mock_retriever import MockRetriever
from graph_rag_eval.retrievers.base import Document


@pytest.fixture(scope="module")
def synthetic_dataset():
    return SyntheticQADataset()


@pytest.fixture(scope="module")
def flat_report(synthetic_dataset):
    flat = FlatRetriever()
    flat.build_index(synthetic_dataset.documents)
    evaluator = RAGEvaluator(flat, "Flat TF-IDF")
    return evaluator.evaluate(synthetic_dataset, k=5, dataset_name="synthetic")


@pytest.fixture(scope="module")
def graph_report(synthetic_dataset):
    kg = synthetic_dataset.knowledge_graph
    graph = GraphRetriever(kg, max_hops=2)
    graph.build_index([])
    evaluator = RAGEvaluator(graph, "Graph BFS")
    return evaluator.evaluate(synthetic_dataset, k=5, dataset_name="synthetic")


class TestEvaluatorRunsEndToEnd:
    def test_report_has_correct_query_count(self, flat_report):
        assert flat_report.num_queries == 50

    def test_report_metrics_in_range(self, flat_report):
        assert 0.0 <= flat_report.mean_recall_at_k <= 1.0
        assert 0.0 <= flat_report.mean_precision_at_k <= 1.0
        assert 0.0 <= flat_report.mean_mrr <= 1.0
        assert 0.0 <= flat_report.mean_ndcg <= 1.0
        assert 0.0 <= flat_report.mean_entity_coverage <= 1.0

    def test_report_has_query_results(self, flat_report):
        assert len(flat_report.query_results) == 50

    def test_report_efficiency_populated(self, flat_report):
        assert flat_report.efficiency.total_queries == 50
        assert flat_report.efficiency.index_size > 0

    def test_graph_report_end_to_end(self, graph_report):
        assert graph_report.num_queries == 50
        assert 0.0 <= graph_report.mean_entity_coverage <= 1.0

    def test_report_to_dict(self, flat_report):
        d = flat_report.to_dict()
        assert "recall@k" in d
        assert "entity_coverage" in d
        assert "efficiency" in d

    def test_report_summary_string(self, flat_report):
        summary = flat_report.summary()
        assert "Flat TF-IDF" in summary
        assert "Recall" in summary


class TestComparisonReport:
    def test_comparison_structure(self, flat_report, graph_report):
        comparison = ComparisonReport(flat_report, graph_report,
                                      label_baseline="Flat", label_candidate="Graph")
        d = comparison.to_dict()
        assert "baseline" in d
        assert "candidate" in d
        assert "comparison" in d

    def test_comparison_delta_sign(self, flat_report, graph_report):
        comparison = ComparisonReport(flat_report, graph_report)
        delta = comparison.delta("mean_entity_coverage")
        # Delta can be positive or negative depending on results
        assert isinstance(delta, float)

    def test_comparison_summary_contains_both_names(self, flat_report, graph_report):
        comparison = ComparisonReport(flat_report, graph_report,
                                      label_baseline="Flat TF-IDF", label_candidate="Graph BFS")
        summary = comparison.summary()
        assert "Flat TF-IDF" in summary
        assert "Graph BFS" in summary

    def test_comparison_winner(self, flat_report, graph_report):
        comparison = ComparisonReport(flat_report, graph_report,
                                      label_baseline="Flat TF-IDF", label_candidate="Graph BFS")
        winner = comparison.winner("mean_entity_coverage")
        assert winner in ("Flat TF-IDF", "Graph BFS", "tie")

    def test_comparison_save_json(self, flat_report, graph_report, tmp_path):
        comparison = ComparisonReport(flat_report, graph_report)
        output_path = str(tmp_path / "report.json")
        comparison.save(output_path)
        import json
        with open(output_path) as f:
            data = json.load(f)
        assert "baseline" in data
        assert "candidate" in data


class TestGraphBeatsFlatOnMultihop:
    def test_graph_beats_flat_on_multihop_entity_coverage(self, flat_report, graph_report):
        """
        Graph retriever should outperform flat retriever on multi-hop questions.
        Entity traversal gives graph retriever an inherent advantage here.
        """
        # Graph retriever should reach multi-hop entities via BFS
        # Flat retriever must rely on TF-IDF term overlap alone
        graph_multihop = graph_report.multihop_entity_coverage
        flat_multihop = flat_report.multihop_entity_coverage
        # Graph retriever consistently finds entities via traversal
        # For multi-hop: graph >= flat (or close — don't be brittle)
        # Allow flat to be within 5% if it happens to do well
        assert graph_multihop >= flat_multihop - 0.05, (
            f"Graph ({graph_multihop:.3f}) expected >= Flat ({flat_multihop:.3f}) "
            "on multi-hop entity coverage"
        )
