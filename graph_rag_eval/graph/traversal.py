"""GraphTraversal: BFS/DFS from query entities, k-hop expansion."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterator, List, Optional, Set, Tuple

from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph


class GraphTraversal:
    """
    BFS and DFS traversal utilities for a KnowledgeGraph.

    Example::

        kg = KnowledgeGraph()
        # ... populate kg ...
        trav = GraphTraversal(kg)
        nodes = trav.bfs("acme", max_hops=2)
        # → [("acme", 0), ("alice", 1), ("globex", 2), ...]
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self._kg = kg

    def bfs(
        self,
        start: str,
        max_hops: int = 2,
        relation: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """
        Breadth-first traversal from *start*.

        Args:
            start: Starting node ID.
            max_hops: Maximum hop depth.
            relation: If set, only traverse edges with this relation type.

        Returns:
            List of (node_id, hop_distance) pairs in BFS order.
        """
        if start not in self._kg.nodes:
            return []

        visited: Set[str] = {start}
        queue: deque[Tuple[str, int]] = deque([(start, 0)])
        result: List[Tuple[str, int]] = []

        while queue:
            node_id, depth = queue.popleft()
            result.append((node_id, depth))

            if depth < max_hops:
                for neighbor in self._kg.get_neighbors(node_id, relation):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return result

    def multi_source_bfs(
        self,
        seeds: List[str],
        max_hops: int = 2,
        relation: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """
        BFS from multiple seed nodes simultaneously.

        Returns nodes in order of minimum hop distance from any seed.
        """
        valid_seeds = [s for s in seeds if s in self._kg.nodes]
        if not valid_seeds:
            return []

        visited: Set[str] = set(valid_seeds)
        queue: deque[Tuple[str, int]] = deque((s, 0) for s in valid_seeds)
        result: List[Tuple[str, int]] = []

        while queue:
            node_id, depth = queue.popleft()
            result.append((node_id, depth))

            if depth < max_hops:
                for neighbor in self._kg.get_neighbors(node_id, relation):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return result

    def dfs(
        self,
        start: str,
        max_hops: int = 2,
        relation: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """
        Depth-first traversal from *start*.

        Returns:
            List of (node_id, depth) pairs in DFS visit order.
        """
        if start not in self._kg.nodes:
            return []

        visited: Set[str] = set()
        result: List[Tuple[str, int]] = []

        def _dfs(node_id: str, depth: int) -> None:
            if node_id in visited or depth > max_hops:
                return
            visited.add(node_id)
            result.append((node_id, depth))
            for neighbor in self._kg.get_neighbors(node_id, relation):
                _dfs(neighbor, depth + 1)

        _dfs(start, 0)
        return result

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        BFS shortest path between source and target.

        Returns:
            List of node IDs forming the path, or None if unreachable.
        """
        if source not in self._kg.nodes or target not in self._kg.nodes:
            return None
        if source == target:
            return [source]

        visited: Set[str] = {source}
        queue: deque[List[str]] = deque([[source]])

        while queue:
            path = queue.popleft()
            node = path[-1]
            for neighbor in self._kg.get_neighbors(node):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def hop_distance(self, source: str, target: str) -> int:
        """
        Minimum hop distance between source and target.

        Returns:
            Number of hops, or -1 if unreachable.
        """
        path = self.shortest_path(source, target)
        if path is None:
            return -1
        return len(path) - 1

    def k_hop_neighborhood(
        self, node_id: str, k: int, relation: Optional[str] = None
    ) -> Set[str]:
        """Return the set of all nodes within k hops of node_id."""
        return {n for n, _ in self.bfs(node_id, max_hops=k, relation=relation)}
