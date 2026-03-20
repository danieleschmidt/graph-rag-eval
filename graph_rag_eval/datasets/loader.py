"""Abstract DatasetLoader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QAPair:
    """
    A single question-answer pair with ground truth retrieval labels.

    Attributes:
        question: The natural language question.
        answer: Expected answer string.
        relevant_ids: IDs of documents/entities relevant to this question.
        answer_entities: Entity IDs directly needed to answer the question.
        answer_relations: (source, target, relation) tuples needed to answer.
        metadata: Optional question metadata (difficulty, question type, etc.).
    """

    question: str
    answer: str
    relevant_ids: List[str] = field(default_factory=list)
    answer_entities: List[str] = field(default_factory=list)
    answer_relations: List[Tuple[str, str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_multihop(self) -> bool:
        """Return True if this question requires multi-hop reasoning."""
        return self.metadata.get("multihop", False)


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    Subclass this to load your own QA datasets into graph-rag-eval.

    Example::

        class MyLoader(DatasetLoader):
            def load(self):
                return [QAPair(question="...", answer="...", relevant_ids=[...])]
    """

    @abstractmethod
    def load(self) -> List[QAPair]:
        """Load and return all QA pairs in the dataset."""

    def __len__(self) -> int:
        return len(self.load())
