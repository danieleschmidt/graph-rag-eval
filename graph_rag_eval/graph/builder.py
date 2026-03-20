"""GraphBuilder: extract entities from text and build a KnowledgeGraph."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from graph_rag_eval.graph.knowledge_graph import KnowledgeGraph


class GraphBuilder:
    """
    Build a KnowledgeGraph from structured or semi-structured text.

    Usage::

        builder = GraphBuilder()
        builder.add_entity("acme", name="Acme Corp", type="company", sector="tech",
                           description="Acme Corp is a tech company based in USA.")
        builder.add_entity("alice", name="Alice", type="person",
                           description="Alice is a software engineer at Acme Corp.")
        builder.add_relation("alice", "acme", "works_at")
        kg = builder.build()
    """

    def __init__(self) -> None:
        self._entities: List[Tuple[str, Dict[str, Any]]] = []
        self._relations: List[Tuple[str, str, str, Dict[str, Any]]] = []

    def add_entity(self, entity_id: str, **attrs: Any) -> "GraphBuilder":
        """Register an entity with attributes."""
        self._entities.append((entity_id, attrs))
        return self

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        bidirectional: bool = True,
        **attrs: Any,
    ) -> "GraphBuilder":
        """Register a relation between two entities."""
        self._relations.append((source, target, relation, {"bidirectional": bidirectional, **attrs}))
        return self

    def build(self) -> KnowledgeGraph:
        """Construct and return the KnowledgeGraph."""
        kg = KnowledgeGraph()
        for entity_id, attrs in self._entities:
            kg.add_node(entity_id, **attrs)
        for source, target, relation, attrs in self._relations:
            bidirectional = attrs.pop("bidirectional", True)
            kg.add_edge(source, target, relation=relation, bidirectional=bidirectional, **attrs)
        return kg

    @staticmethod
    def extract_entities_from_text(
        text: str,
        entity_patterns: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Extract (entity_id, entity_type) pairs from text using regex patterns.

        Args:
            text: Input text to scan.
            entity_patterns: Mapping of entity_type → regex pattern.
                             Default patterns match capitalized multi-word phrases.

        Returns:
            List of (matched_text, entity_type) tuples.
        """
        if entity_patterns is None:
            entity_patterns = {
                "organization": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
                "person": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
            }

        found: List[Tuple[str, str]] = []
        seen: set = set()
        for entity_type, pattern in entity_patterns.items():
            for match in re.finditer(pattern, text):
                mention = match.group(0).strip()
                if mention not in seen:
                    seen.add(mention)
                    found.append((mention, entity_type))
        return found
