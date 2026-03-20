"""Tests for KnowledgeGraph."""

import pytest
from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph


def make_simple_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node("acme", name="Acme Corp", type="company", sector="tech")
    kg.add_node("alice", name="Alice", type="person")
    kg.add_node("globex", name="Globex", type="company", sector="finance")
    kg.add_edge("alice", "acme", relation="works_at")
    kg.add_edge("acme", "globex", relation="partnered_with", bidirectional=False)
    return kg


class TestKnowledgeGraphBuild:
    def test_add_nodes(self):
        kg = KnowledgeGraph()
        kg.add_node("n1", name="Node1")
        kg.add_node("n2", name="Node2")
        assert kg.node_count() == 2
        assert kg.has_node("n1")
        assert kg.has_node("n2")

    def test_node_attributes(self):
        kg = KnowledgeGraph()
        kg.add_node("acme", name="Acme Corp", sector="tech")
        node = kg.get_node("acme")
        assert node["name"] == "Acme Corp"
        assert node["sector"] == "tech"

    def test_add_edge_builds_adjacency(self):
        kg = make_simple_kg()
        neighbors = kg.get_neighbors("alice")
        assert "acme" in neighbors

    def test_bidirectional_edge(self):
        kg = KnowledgeGraph()
        kg.add_node("a")
        kg.add_node("b")
        kg.add_edge("a", "b", relation="linked", bidirectional=True)
        assert "b" in kg.get_neighbors("a")
        assert "a" in kg.get_neighbors("b")

    def test_unidirectional_edge(self):
        kg = KnowledgeGraph()
        kg.add_node("a")
        kg.add_node("b")
        kg.add_edge("a", "b", relation="linked", bidirectional=False)
        assert "b" in kg.get_neighbors("a")
        assert "a" not in kg.get_neighbors("b")

    def test_edge_count(self):
        kg = make_simple_kg()
        # alice-acme bidirectional = 2, acme-globex unidirectional = 1
        assert kg.edge_count() == 3

    def test_get_relations(self):
        kg = make_simple_kg()
        rels = kg.get_relations("alice", "acme")
        assert "works_at" in rels

    def test_missing_node_raises(self):
        kg = KnowledgeGraph()
        kg.add_node("a")
        with pytest.raises(ValueError):
            kg.add_edge("a", "nonexistent", relation="rel")

    def test_update_node_attributes(self):
        kg = KnowledgeGraph()
        kg.add_node("n1", color="red")
        kg.add_node("n1", size=10)  # update
        node = kg.get_node("n1")
        assert node["color"] == "red"
        assert node["size"] == 10

    def test_description_for(self):
        kg = KnowledgeGraph()
        kg.add_node("acme", name="Acme Corp", description="Acme is a tech giant.")
        assert kg.description_for("acme") == "Acme is a tech giant."

    def test_stats(self):
        kg = make_simple_kg()
        stats = kg.stats()
        assert stats["nodes"] == 3
        assert "edges" in stats
