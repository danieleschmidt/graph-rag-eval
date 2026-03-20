"""MockRetriever for testing — returns predetermined documents."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from graph_rag_eval.retrievers.base import BaseRetriever, Document


class MockRetriever(BaseRetriever):
    """
    Retriever that returns a predetermined set of documents per query.

    Useful for unit testing metrics without needing a real retrieval system.

    Usage::

        retriever = MockRetriever(default_docs=[
            Document(id="doc1", content="Acme Corp is in tech."),
            Document(id="doc2", content="Bob works at Acme Corp."),
        ])
        results = retriever.retrieve("Who works at Acme?", k=2)
    """

    def __init__(
        self,
        default_docs: Optional[List[Document]] = None,
        query_map: Optional[Dict[str, List[Document]]] = None,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__()
        self._default_docs: List[Document] = default_docs or []
        self._query_map: Dict[str, List[Document]] = query_map or {}
        self._latency_ms = latency_ms
        self._call_count = 0

    def build_index(self, documents: List[Document]) -> None:
        self._default_docs = list(documents)
        self._index_size = len(documents)

    def _retrieve_impl(self, query: str, k: int) -> List[Document]:
        self._call_count += 1
        self._last_nodes_visited = 0
        docs = self._query_map.get(query, self._default_docs)
        return docs[:k]

    @property
    def call_count(self) -> int:
        """Number of times retrieve() has been called."""
        return self._call_count
