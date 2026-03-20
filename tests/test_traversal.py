"""Tests for GraphTraversal BFS/DFS."""

import pytest
from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph
from graph_rag_eval.graph.traversal import GraphTraversal


def make_chain_kg() -> KnowledgeGraph:
    """A → B → C → D (linear chain)."""
    kg = KnowledgeGraph()
    for node in ["a", "b", "c", "d"]:
        kg.add_node(node, name=node.upper())
    kg.add_edge("a", "b", bidirectional=False)
    kg.add_edge("b", "c", bidirectional=False)
    kg.add_edge("c", "d", bidirectional=False)
    return kg


def make_hub_kg() -> KnowledgeGraph:
    """Hub: center connects to 4 spokes."""
    kg = KnowledgeGraph()
    kg.add_node("center", name="Center")
    for i in range(4):
        kg.add_node(f"spoke{i}", name=f"Spoke{i}")
        kg.add_edge("center", f"spoke{i}", relation="connects", bidirectional=True)
    return kg


class TestBFSTraversalDepth:
    def test_1hop_from_a(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        result = trav.bfs("a", max_hops=1)
        node_ids = {n for n, _ in result}
        assert "a" in node_ids
        assert "b" in node_ids
        assert "c" not in node_ids  # 2 hops away

    def test_2hop_from_a(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        result = trav.bfs("a", max_hops=2)
        node_ids = {n for n, _ in result}
        assert "a" in node_ids
        assert "b" in node_ids
        assert "c" in node_ids
        assert "d" not in node_ids  # 3 hops

    def test_bfs_hop_distances_correct(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        result = dict(trav.bfs("a", max_hops=3))
        assert result["a"] == 0
        assert result["b"] == 1
        assert result["c"] == 2
        assert result["d"] == 3

    def test_bfs_hub(self):
        kg = make_hub_kg()
        trav = GraphTraversal(kg)
        result = trav.bfs("center", max_hops=1)
        node_ids = {n for n, _ in result}
        assert "center" in node_ids
        for i in range(4):
            assert f"spoke{i}" in node_ids

    def test_dfs_visits_all_within_depth(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        result = trav.dfs("a", max_hops=3)
        node_ids = {n for n, _ in result}
        assert node_ids == {"a", "b", "c", "d"}

    def test_multi_source_bfs(self):
        kg = make_hub_kg()
        trav = GraphTraversal(kg)
        result = trav.multi_source_bfs(["spoke0", "spoke1"], max_hops=1)
        node_ids = {n for n, _ in result}
        assert "center" in node_ids

    def test_shortest_path(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        path = trav.shortest_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_shortest_path_unreachable(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        # d has no outgoing edges
        path = trav.shortest_path("d", "a")
        assert path is None

    def test_hop_distance(self):
        kg = make_chain_kg()
        trav = GraphTraversal(kg)
        assert trav.hop_distance("a", "c") == 2
        assert trav.hop_distance("a", "a") == 0
        assert trav.hop_distance("d", "a") == -1

    def test_k_hop_neighborhood(self):
        kg = make_hub_kg()
        trav = GraphTraversal(kg)
        hood = trav.k_hop_neighborhood("spoke0", k=1)
        assert "center" in hood
