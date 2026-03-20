"""Abstract base retriever interface for graph-rag-eval."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A retrieved document/entity."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def __repr__(self) -> str:
        snippet = self.content[:60].replace("\n", " ")
        return f"Document(id={self.id!r}, score={self.score:.3f}, content={snippet!r}...)"


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.

    Subclass this to plug your own retrieval system into graph-rag-eval.

    Example::

        class MyRetriever(BaseRetriever):
            def build_index(self, documents):
                ...

            def retrieve(self, query: str, k: int = 5) -> List[Document]:
                ...
    """

    def __init__(self) -> None:
        self._index_size: int = 0
        self._last_latency_ms: float = 0.0
        self._last_nodes_visited: int = 0

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """Build internal index from a list of documents."""

    @abstractmethod
    def _retrieve_impl(self, query: str, k: int) -> List[Document]:
        """Internal retrieve — override this, not retrieve()."""

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k documents for *query*.

        Wraps _retrieve_impl to record latency.

        Args:
            query: Natural language query string.
            k: Number of documents to return.

        Returns:
            Ordered list of Documents (highest score first).
        """
        start = time.perf_counter()
        results = self._retrieve_impl(query, k)
        self._last_latency_ms = (time.perf_counter() - start) * 1000
        return results

    @property
    def index_size(self) -> int:
        """Number of documents in the index."""
        return self._index_size

    @property
    def last_latency_ms(self) -> float:
        """Latency of the most recent retrieve() call in milliseconds."""
        return self._last_latency_ms

    @property
    def last_nodes_visited(self) -> int:
        """Nodes visited during the most recent retrieve() call (graph traversal)."""
        return self._last_nodes_visited
