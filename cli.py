#!/usr/bin/env python3
"""
graph-rag-eval CLI

Usage:
    graph-rag-eval compare --dataset synthetic --k 5 --output report.json
    graph-rag-eval compare --dataset synthetic --k 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare flat vs graph retriever on a dataset."""
    from graph_rag_eval.datasets.synthetic import SyntheticQADataset
    from graph_rag_eval.evaluator import RAGEvaluator
    from graph_rag_eval.comparison import ComparisonReport
    from graph_rag_eval.retrievers.flat_retriever import FlatRetriever
    from graph_rag_eval.retrievers.graph_retriever import GraphRetriever

    print(f"Loading dataset: {args.dataset}")

    if args.dataset == "synthetic":
        dataset = SyntheticQADataset()
    else:
        print(f"Unknown dataset: {args.dataset!r}. Available: synthetic", file=sys.stderr)
        return 1

    docs = dataset.documents
    kg = dataset.knowledge_graph

    print(f"Dataset loaded: {len(dataset.load())} Q&A pairs, {len(docs)} documents")
    print(f"Knowledge graph: {kg.stats()}")
    print()

    # Build flat retriever
    print("Building flat TF-IDF retriever...")
    flat = FlatRetriever()
    flat.build_index(docs)

    # Build graph retriever
    print(f"Building graph retriever (max_hops={args.max_hops})...")
    graph = GraphRetriever(kg, max_hops=args.max_hops)
    graph.build_index([])

    # Evaluate
    print(f"\nEvaluating flat retriever (k={args.k})...")
    flat_eval = RAGEvaluator(flat, "Flat TF-IDF")
    flat_report = flat_eval.evaluate(dataset, k=args.k, dataset_name=args.dataset,
                                     verbose=args.verbose)

    print(f"\nEvaluating graph retriever (k={args.k})...")
    graph_eval = RAGEvaluator(graph, "Graph (BFS)")
    graph_report = graph_eval.evaluate(dataset, k=args.k, dataset_name=args.dataset,
                                       verbose=args.verbose)

    # Compare
    comparison = ComparisonReport(
        baseline=flat_report,
        candidate=graph_report,
        label_baseline="Flat TF-IDF",
        label_candidate="Graph BFS",
    )

    print()
    print(comparison.summary())

    if args.output:
        comparison.save(args.output)

    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graph-rag-eval",
        description="Evaluation framework for graph-augmented RAG vs flat retrieval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare flat and graph retrievers on a dataset.",
    )
    compare_parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic"],
        help="Dataset to evaluate on (default: synthetic)",
    )
    compare_parser.add_argument(
        "--k", type=int, default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    compare_parser.add_argument(
        "--max-hops", type=int, default=2, dest="max_hops",
        help="Max BFS hops for graph retriever (default: 2)",
    )
    compare_parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file for the comparison report",
    )
    compare_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-query results",
    )
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
