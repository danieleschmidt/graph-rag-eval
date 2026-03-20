"""Graph-specific metrics: entity coverage, relation coverage, hop distance."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from graph_rag_eval.retrievers.base import Document


def entity_coverage(
    retrieved_docs: List[Document],
    answer_entities: List[str],
) -> float:
    """
    Fraction of answer entities found in the retrieved documents.

    Checks whether an answer entity appears in any retrieved document's content
    or metadata (by entity ID or name).

    Args:
        retrieved_docs: Documents returned by the retriever.
        answer_entities: Entity IDs or names required to answer the question.

    Returns:
        Entity coverage in [0, 1].
    """
    if not answer_entities:
        return 0.0

    # Build a set of entity mentions from retrieved docs
    retrieved_ids = {doc.id for doc in retrieved_docs}
    retrieved_content = " ".join(doc.content.lower() for doc in retrieved_docs)

    found = 0
    for entity in answer_entities:
        entity_lower = entity.lower()
        if entity in retrieved_ids or entity_lower in retrieved_content:
            found += 1
    return found / len(answer_entities)


def relation_coverage(
    retrieved_docs: List[Document],
    answer_relations: List[Tuple[str, str, str]],
) -> float:
    """
    Fraction of answer relations whose endpoints are both in retrieved docs.

    A relation (source, target, relation_type) is "covered" if both source and
    target entities appear in the retrieved documents.

    Args:
        retrieved_docs: Documents returned by the retriever.
        answer_relations: List of (source_id, target_id, relation_type) tuples
                          needed to fully answer the question.

    Returns:
        Relation coverage in [0, 1].
    """
    if not answer_relations:
        return 0.0

    retrieved_ids = {doc.id for doc in retrieved_docs}
    retrieved_names = set()
    for doc in retrieved_docs:
        name = doc.metadata.get("name", "")
        if name:
            retrieved_names.add(name.lower())

    def entity_in_retrieved(entity_id: str) -> bool:
        if entity_id in retrieved_ids:
            return True
        # Check metadata names
        for doc in retrieved_docs:
            if doc.metadata.get("name", "").lower() == entity_id.lower():
                return True
            if entity_id.lower() in doc.content.lower():
                return True
        return False

    covered = sum(
        1
        for src, tgt, _ in answer_relations
        if entity_in_retrieved(src) and entity_in_retrieved(tgt)
    )
    return covered / len(answer_relations)


def avg_hop_distance(
    retrieved_docs: List[Document],
    answer_entities: List[str],
) -> float:
    """
    Average hop distance from seed entities to answer entities in retrieved docs.

    Uses the 'hop_distance' metadata field set by GraphRetriever.
    Returns -1 if no answer entities are in retrieved docs.

    Args:
        retrieved_docs: Documents returned by GraphRetriever (with hop_distance metadata).
        answer_entities: Entity IDs required to answer the question.

    Returns:
        Average hop distance, or -1.0 if no answer entities found.
    """
    answer_set = set(answer_entities)
    hops = []
    for doc in retrieved_docs:
        if doc.id in answer_set or doc.metadata.get("name", "") in answer_set:
            hop = doc.metadata.get("hop_distance")
            if hop is not None:
                hops.append(hop)

    if not hops:
        return -1.0
    return sum(hops) / len(hops)


def hop_efficiency_score(avg_hop: float, max_hops: int = 2) -> float:
    """
    Normalize avg_hop_distance to [0, 1] where 1 = all answers at hop 0.

    Args:
        avg_hop: Output of avg_hop_distance().
        max_hops: Maximum hop depth used during retrieval.

    Returns:
        Efficiency score in [0, 1], or 0.0 if avg_hop is -1.
    """
    if avg_hop < 0:
        return 0.0
    return max(0.0, 1.0 - avg_hop / max_hops)
