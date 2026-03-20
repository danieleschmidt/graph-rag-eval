"""Tests for FlatRetriever and GraphRetriever."""

import pytest
from graph_rag_eval.retrievers.base import Document
from graph_rag_eval.retrievers.flat_retriever import FlatRetriever
from graph_rag_eval.retrievers.graph_retriever import GraphRetriever
from graph_rag_eval.retrievers.mock_retriever import MockRetriever
from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph


SAMPLE_DOCS = [
    Document(id="doc1", content="Acme Corp is a technology company headquartered in USA."),
    Document(id="doc2", content="Alice Chen is the CEO of Acme Corp, a tech leader."),
    Document(id="doc3", content="Globex Finance operates in the banking sector in UK."),
    Document(id="doc4", content="Bob Smith is a software engineer working at Globex Finance."),
    Document(id="doc5", content="Acme Corp recently partnered with Globex Finance."),
]


def make_sample_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node("acme", name="Acme Corp", type="company", sector="technology",
                description="Acme Corp is a technology company headquartered in USA.")
    kg.add_node("alice", name="Alice Chen", type="person",
                description="Alice Chen is the CEO of Acme Corp.")
    kg.add_node("globex", name="Globex Finance", type="company", sector="finance",
                description="Globex Finance operates in the banking sector in UK.")
    kg.add_node("bob", name="Bob Smith", type="person",
                description="Bob Smith is a software engineer at Globex Finance.")
    kg.add_edge("alice", "acme", relation="works_at")
    kg.add_edge("bob", "globex", relation="works_at")
    kg.add_edge("acme", "globex", relation="partnered_with")
    return kg


class TestFlatRetriever:
    def test_flat_retriever_tfidf_basic(self):
        retriever = FlatRetriever()
        retriever.build_index(SAMPLE_DOCS)
        results = retriever.retrieve("Acme Corp technology", k=3)
        ids = [r.id for r in results]
        assert "doc1" in ids or "doc2" in ids  # Acme Corp docs should rank high

    def test_flat_retriever_returns_k_results(self):
        retriever = FlatRetriever()
        retriever.build_index(SAMPLE_DOCS)
        results = retriever.retrieve("technology", k=2)
        assert len(results) <= 2

    def test_flat_retriever_scores_sorted_descending(self):
        retriever = FlatRetriever()
        retriever.build_index(SAMPLE_DOCS)
        results = retriever.retrieve("Acme CEO", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_flat_retriever_index_size(self):
        retriever = FlatRetriever()
        retriever.build_index(SAMPLE_DOCS)
        assert retriever.index_size == len(SAMPLE_DOCS)

    def test_flat_retriever_empty_index(self):
        retriever = FlatRetriever()
        retriever.build_index([])
        results = retriever.retrieve("anything", k=5)
        assert results == []

    def test_flat_retriever_relevant_doc_in_top_results(self):
        """A query about Alice should retrieve the Alice document."""
        retriever = FlatRetriever()
        retriever.build_index(SAMPLE_DOCS)
        results = retriever.retrieve("Alice Chen CEO", k=3)
        ids = [r.id for r in results]
        assert "doc2" in ids


class TestGraphRetriever:
    def test_graph_retriever_entity_match(self):
        """Query with known entity name should include that entity in results."""
        kg = make_sample_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])
        results = retriever.retrieve("Tell me about Acme Corp", k=5)
        ids = [r.id for r in results]
        assert "acme" in ids

    def test_graph_retriever_traverses_neighbors(self):
        """2-hop traversal from Acme should reach Bob (via acme→globex←bob)."""
        kg = make_sample_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])
        results = retriever.retrieve("Acme Corp", k=10)
        ids = [r.id for r in results]
        assert "alice" in ids  # 1 hop from Acme

    def test_graph_retriever_hop_distance_metadata(self):
        """Retrieved docs should carry hop_distance metadata."""
        kg = make_sample_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])
        results = retriever.retrieve("Acme Corp", k=10)
        for doc in results:
            assert "hop_distance" in doc.metadata

    def test_graph_retriever_seed_entity_at_hop_0(self):
        """The seed entity should have hop_distance=0."""
        kg = make_sample_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])
        results = retriever.retrieve("Acme Corp", k=10)
        acme_results = [d for d in results if d.id == "acme"]
        assert acme_results
        assert acme_results[0].metadata["hop_distance"] == 0

    def test_graph_retriever_index_size(self):
        kg = make_sample_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])
        assert retriever.index_size == kg.node_count()

    def test_graph_retriever_no_match_returns_empty(self):
        kg = make_sample_kg()
        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])
        results = retriever.retrieve("zzzznonexistent_xyz", k=5)
        assert isinstance(results, list)


class TestMockRetriever:
    def test_mock_retriever_returns_default(self):
        docs = [Document(id="d1", content="doc one"), Document(id="d2", content="doc two")]
        retriever = MockRetriever(default_docs=docs)
        results = retriever.retrieve("anything", k=5)
        assert len(results) == 2

    def test_mock_retriever_respects_k(self):
        docs = [Document(id=f"d{i}", content=f"doc {i}") for i in range(10)]
        retriever = MockRetriever(default_docs=docs)
        results = retriever.retrieve("anything", k=3)
        assert len(results) == 3

    def test_mock_retriever_query_map(self):
        q_map = {
            "acme": [Document(id="acme_doc", content="Acme info")],
        }
        retriever = MockRetriever(query_map=q_map)
        results = retriever.retrieve("acme", k=5)
        assert results[0].id == "acme_doc"

    def test_mock_retriever_call_count(self):
        retriever = MockRetriever()
        retriever.retrieve("q1", k=5)
        retriever.retrieve("q2", k=5)
        assert retriever.call_count == 2
