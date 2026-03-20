"""TF-IDF flat retriever — no embedding model needed."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from graph_rag_eval.retrievers.base import BaseRetriever, Document


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


class FlatRetriever(BaseRetriever):
    """
    TF-IDF retriever over a flat list of documents.

    Uses pure-Python TF-IDF with cosine similarity — no scikit-learn or
    embedding model required.

    Usage::

        retriever = FlatRetriever()
        retriever.build_index(documents)
        results = retriever.retrieve("Who works at Acme?", k=5)
    """

    def __init__(self) -> None:
        super().__init__()
        self._documents: List[Document] = []
        self._tfidf_matrix: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}

    def build_index(self, documents: List[Document]) -> None:
        """Build TF-IDF index from documents."""
        self._documents = list(documents)
        self._index_size = len(documents)

        # Count document frequencies
        df: Dict[str, int] = defaultdict(int)
        for doc in self._documents:
            tokens = set(_tokenize(doc.content))
            for t in tokens:
                df[t] += 1

        n = len(self._documents)
        self._idf = {
            term: math.log((n + 1) / (freq + 1)) + 1  # smoothed IDF
            for term, freq in df.items()
        }

        # Build TF-IDF vectors
        self._tfidf_matrix = []
        for doc in self._documents:
            tokens = _tokenize(doc.content)
            tf: Dict[str, float] = defaultdict(float)
            for t in tokens:
                tf[t] += 1
            total = len(tokens) or 1
            vec: Dict[str, float] = {
                t: (count / total) * self._idf.get(t, 1.0)
                for t, count in tf.items()
            }
            self._tfidf_matrix.append(vec)

    def _retrieve_impl(self, query: str, k: int) -> List[Document]:
        if not self._documents:
            return []

        # Build query TF-IDF vector
        tokens = _tokenize(query)
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = len(tokens) or 1
        q_vec: Dict[str, float] = {
            t: (count / total) * self._idf.get(t, 1.0)
            for t, count in tf.items()
        }

        # Cosine similarity
        scores: List[Tuple[int, float]] = []
        for idx, doc_vec in enumerate(self._tfidf_matrix):
            dot = sum(q_vec.get(t, 0) * w for t, w in doc_vec.items())
            q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1e-9
            d_norm = math.sqrt(sum(v ** 2 for v in doc_vec.values())) or 1e-9
            scores.append((idx, dot / (q_norm * d_norm)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]

        results = []
        for idx, score in top_k:
            doc = self._documents[idx]
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=dict(doc.metadata),
                    score=score,
                )
            )
        return results
