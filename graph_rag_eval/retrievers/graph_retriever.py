"""Graph-aware retriever using entity-linked BFS traversal."""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph
from graph_rag_eval.retrievers.base import BaseRetriever, Document


class GraphRetriever(BaseRetriever):
    """
    Retrieves documents via entity linking + BFS expansion over a KnowledgeGraph.

    Algorithm:
        1. Extract entity mentions from the query (regex match against KG entity names).
        2. BFS from matched entities up to *max_hops* hops.
        3. Collect entity descriptions as Documents, ranked by hop distance.

    Usage::

        kg = KnowledgeGraph()
        kg.add_node("acme", "Acme Corp is a tech company based in USA.")
        # ... add more nodes/edges ...

        retriever = GraphRetriever(kg, max_hops=2)
        retriever.build_index([])  # KG is the index — call still required
        results = retriever.retrieve("Who works at Acme Corp?", k=5)
    """

    def __init__(self, kg: KnowledgeGraph, max_hops: int = 2) -> None:
        super().__init__()
        self._kg = kg
        self._max_hops = max_hops
        # id → Document mapping built in build_index
        self._doc_map: Dict[str, Document] = {}

    def build_index(self, documents: Optional[List[Document]] = None) -> None:
        """
        Build index from the KnowledgeGraph nodes.

        The *documents* argument is ignored — the KG itself is the index.
        Pass an empty list or None when the KG is already populated.
        """
        self._doc_map = {}
        for node_id, node_data in self._kg.nodes.items():
            description = node_data.get("description", "")
            # Build a rich text description from node attributes
            attrs = {k: v for k, v in node_data.items() if k != "description"}
            attr_str = "; ".join(f"{k}={v}" for k, v in attrs.items())
            content = description or (f"{node_id} ({attr_str})" if attrs else node_id)
            self._doc_map[node_id] = Document(
                id=node_id,
                content=content,
                metadata=dict(node_data),
            )
        self._index_size = len(self._doc_map)

    def _extract_entity_mentions(self, query: str) -> List[str]:
        """Return node IDs whose names appear in the query (case-insensitive)."""
        query_lower = query.lower()
        matched = []
        # Try matching by node 'name' attribute first, then by node_id
        for node_id, node_data in self._kg.nodes.items():
            name = node_data.get("name", node_id)
            if re.search(re.escape(name.lower()), query_lower):
                matched.append(node_id)
            elif re.search(re.escape(node_id.lower()), query_lower):
                if node_id not in matched:
                    matched.append(node_id)
        return matched

    def _bfs(self, seeds: List[str], max_hops: int) -> List[Tuple[str, int]]:
        """BFS from seed nodes; returns (node_id, hop_distance) pairs."""
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()
        result: List[Tuple[str, int]] = []

        for seed in seeds:
            if seed in self._kg.nodes and seed not in visited:
                queue.append((seed, 0))
                visited.add(seed)

        while queue:
            node_id, depth = queue.popleft()
            result.append((node_id, depth))
            self._last_nodes_visited += 1

            if depth < max_hops:
                for neighbor in self._kg.get_neighbors(node_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return result

    def _retrieve_impl(self, query: str, k: int) -> List[Document]:
        self._last_nodes_visited = 0

        seeds = self._extract_entity_mentions(query)

        # Fallback: if no entities matched, try partial token matching
        if not seeds:
            query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
            for node_id, node_data in self._kg.nodes.items():
                name_tokens = set(
                    re.findall(r"[a-z0-9]+", node_data.get("name", node_id).lower())
                )
                if query_tokens & name_tokens:
                    seeds.append(node_id)

        if not seeds:
            return []

        traversal = self._bfs(seeds, self._max_hops)

        # Score: closer hop = higher score
        docs: List[Document] = []
        for node_id, hop in traversal:
            if node_id in self._doc_map:
                doc = self._doc_map[node_id]
                score = 1.0 / (1.0 + hop)  # hop=0 → 1.0, hop=1 → 0.5, hop=2 → 0.33
                docs.append(
                    Document(
                        id=doc.id,
                        content=doc.content,
                        metadata={**doc.metadata, "hop_distance": hop},
                        score=score,
                    )
                )

        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[:k]
