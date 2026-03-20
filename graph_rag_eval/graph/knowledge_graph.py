"""KnowledgeGraph: nodes (entities), edges (relations), adjacency list."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


class KnowledgeGraph:
    """
    A simple in-memory knowledge graph.

    Nodes represent entities (companies, people, etc.).
    Edges represent typed relations between entities.

    Example::

        kg = KnowledgeGraph()
        kg.add_node("acme", name="Acme Corp", type="company", sector="tech")
        kg.add_node("alice", name="Alice", type="person")
        kg.add_edge("alice", "acme", relation="works_at")

        neighbors = kg.get_neighbors("alice")
        # → ["acme"]
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # adjacency list: source → list of (target, relation, edge_attrs)
        self._adj: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        # reverse adjacency for bidirectional traversal
        self._rev_adj: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
        self._edge_count: int = 0

    # -------------------------------------------------------------------------
    # Node operations
    # -------------------------------------------------------------------------

    def add_node(self, node_id: str, **attrs: Any) -> None:
        """Add or update a node with optional attributes."""
        if node_id in self._nodes:
            self._nodes[node_id].update(attrs)
        else:
            self._nodes[node_id] = dict(attrs)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return node attributes or None if not found."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> bool:
        """Remove node and all its incident edges."""
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Remove outgoing edges
        self._adj.pop(node_id, None)
        # Remove incoming edges
        self._rev_adj.pop(node_id, None)
        # Clean up other adjacency lists
        for src in list(self._adj.keys()):
            self._adj[src] = [
                (tgt, rel, attrs)
                for tgt, rel, attrs in self._adj[src]
                if tgt != node_id
            ]
        for tgt in list(self._rev_adj.keys()):
            self._rev_adj[tgt] = [
                (src, rel, attrs)
                for src, rel, attrs in self._rev_adj[tgt]
                if src != node_id
            ]
        return True

    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        """Read-only view of all nodes."""
        return self._nodes

    def node_count(self) -> int:
        return len(self._nodes)

    # -------------------------------------------------------------------------
    # Edge operations
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        bidirectional: bool = True,
        **attrs: Any,
    ) -> None:
        """
        Add a directed edge from *source* to *target*.

        Args:
            source: Source node ID (must already exist).
            target: Target node ID (must already exist).
            relation: Edge type label.
            bidirectional: Also add reverse edge (default True).
            **attrs: Additional edge attributes.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node {source!r} not in graph")
        if target not in self._nodes:
            raise ValueError(f"Target node {target!r} not in graph")

        edge_attrs = dict(attrs)
        self._adj[source].append((target, relation, edge_attrs))
        self._rev_adj[target].append((source, relation, edge_attrs))
        self._edge_count += 1

        if bidirectional:
            self._adj[target].append((source, relation, edge_attrs))
            self._rev_adj[source].append((target, relation, edge_attrs))
            self._edge_count += 1

    def get_edges(
        self, source: str, relation: Optional[str] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return outgoing edges from *source*, optionally filtered by relation."""
        edges = self._adj.get(source, [])
        if relation:
            edges = [(t, r, a) for t, r, a in edges if r == relation]
        return edges

    def get_neighbors(
        self, node_id: str, relation: Optional[str] = None
    ) -> List[str]:
        """Return IDs of neighbors reachable from *node_id*."""
        return [t for t, r, _ in self.get_edges(node_id, relation)]

    def get_relations(self, source: str, target: str) -> List[str]:
        """Return all relation types between source and target."""
        return [r for t, r, _ in self._adj.get(source, []) if t == target]

    def edge_count(self) -> int:
        return self._edge_count

    def edges(self) -> Iterator[Tuple[str, str, str, Dict[str, Any]]]:
        """Iterate over all (source, target, relation, attrs) tuples."""
        for src, edge_list in self._adj.items():
            for tgt, rel, attrs in edge_list:
                yield src, tgt, rel, attrs

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def description_for(self, node_id: str) -> str:
        """Return human-readable description of a node."""
        node = self._nodes.get(node_id)
        if node is None:
            return node_id
        desc = node.get("description", "")
        if desc:
            return desc
        name = node.get("name", node_id)
        attrs = {k: v for k, v in node.items() if k not in ("name", "description")}
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        return f"{name} ({attr_str})" if attr_str else name

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": self._edge_count,
        }

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={len(self._nodes)}, edges={self._edge_count})"
