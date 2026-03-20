"""Tests for retrieval and graph-specific metrics."""

import math
import pytest
from graph_rag_eval.metrics.retrieval import recall_at_k, precision_at_k, mrr, ndcg
from graph_rag_eval.metrics.answer_quality import exact_match, f1_overlap, answer_coverage
from graph_rag_eval.metrics.graph_specific import entity_coverage, relation_coverage, avg_hop_distance
from graph_rag_eval.retrievers.base import Document


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["a", "b"], ["x", "y", "z"], k=3) == 0.0

    def test_partial_recall(self):
        score = recall_at_k(["a", "b", "c"], ["a", "x", "b"], k=3)
        assert abs(score - 2 / 3) < 1e-9

    def test_cutoff_k_matters(self):
        # "b" is at position 4, beyond k=3
        score = recall_at_k(["a", "b"], ["a", "x", "y", "b"], k=3)
        assert score == 0.5  # only "a" found in top 3

    def test_empty_relevant(self):
        assert recall_at_k([], ["a", "b"], k=5) == 0.0

    def test_recall_at_k_equals_5(self):
        # exactly 5 relevant docs, all in top 5
        relevant = ["a", "b", "c", "d", "e"]
        retrieved = ["a", "b", "c", "d", "e"]
        assert recall_at_k(relevant, retrieved, k=5) == 1.0


class TestNDCG:
    def test_perfect_ndcg(self):
        relevant = ["a", "b", "c"]
        retrieved = ["a", "b", "c", "d", "e"]
        score = ndcg(relevant, retrieved, k=5)
        assert score == pytest.approx(1.0)

    def test_zero_ndcg(self):
        score = ndcg(["a", "b"], ["x", "y", "z"], k=5)
        assert score == 0.0

    def test_ndcg_penalizes_lower_rank(self):
        """Relevant doc at rank 1 should score higher than at rank 3."""
        relevant = ["a"]
        rank1 = ndcg(relevant, ["a", "b", "c"], k=3)
        rank3 = ndcg(relevant, ["b", "c", "a"], k=3)
        assert rank1 > rank3

    def test_ndcg_empty_relevant(self):
        assert ndcg([], ["a", "b"], k=5) == 0.0


class TestMRR:
    def test_mrr_first_position(self):
        assert mrr(["a"], ["a", "b", "c"]) == 1.0

    def test_mrr_third_position(self):
        assert mrr(["a"], ["x", "y", "a"]) == pytest.approx(1 / 3)

    def test_mrr_not_found(self):
        assert mrr(["a"], ["x", "y", "z"]) == 0.0


class TestEntityCoverage:
    def _docs(self, ids):
        return [Document(id=i, content=f"Entity {i}") for i in ids]

    def test_full_entity_coverage(self):
        docs = self._docs(["alice", "acme"])
        score = entity_coverage(docs, ["alice", "acme"])
        assert score == 1.0

    def test_partial_entity_coverage(self):
        docs = self._docs(["alice"])
        score = entity_coverage(docs, ["alice", "acme"])
        assert score == 0.5

    def test_zero_entity_coverage(self):
        docs = self._docs(["globex"])
        score = entity_coverage(docs, ["alice", "acme"])
        assert score == 0.0

    def test_empty_answer_entities(self):
        docs = self._docs(["alice"])
        score = entity_coverage(docs, [])
        assert score == 0.0


class TestRelationCoverage:
    def _docs(self, ids):
        return [Document(id=i, content=f"Entity {i}") for i in ids]

    def test_full_relation_coverage(self):
        docs = self._docs(["alice", "acme"])
        score = relation_coverage(docs, [("alice", "acme", "works_at")])
        assert score == 1.0

    def test_partial_relation_coverage(self):
        docs = self._docs(["alice"])  # only alice, not acme
        score = relation_coverage(docs, [("alice", "acme", "works_at")])
        assert score == 0.0  # both endpoints needed

    def test_empty_answer_relations(self):
        docs = self._docs(["alice"])
        score = relation_coverage(docs, [])
        assert score == 0.0


class TestAnswerQuality:
    def test_exact_match_equal(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_exact_match_case_insensitive(self):
        assert exact_match("paris", "PARIS") == 1.0

    def test_exact_match_different(self):
        assert exact_match("Paris", "London") == 0.0

    def test_f1_overlap_partial(self):
        score = f1_overlap("Alice works at Acme", "Alice Acme Corp")
        assert 0 < score < 1.0

    def test_f1_perfect(self):
        assert f1_overlap("hello world", "hello world") == 1.0

    def test_answer_coverage(self):
        texts = ["Alice works at Acme Corp", "Acme Corp is a tech company"]
        score = answer_coverage(texts, "Acme Corp tech")
        assert score > 0.5


class TestAvgHopDistance:
    def _doc(self, doc_id, hop):
        return Document(id=doc_id, content=f"Entity {doc_id}",
                        metadata={"hop_distance": hop})

    def test_avg_hop_all_at_zero(self):
        docs = [self._doc("alice", 0), self._doc("acme", 0)]
        score = avg_hop_distance(docs, ["alice", "acme"])
        assert score == 0.0

    def test_avg_hop_mixed(self):
        docs = [self._doc("alice", 0), self._doc("globex", 2)]
        score = avg_hop_distance(docs, ["alice", "globex"])
        assert score == 1.0  # (0 + 2) / 2

    def test_avg_hop_not_found(self):
        docs = [self._doc("alice", 0)]
        score = avg_hop_distance(docs, ["nobody"])
        assert score == -1.0
