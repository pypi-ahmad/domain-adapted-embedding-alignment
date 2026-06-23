"""Retrieval metric implementations for baseline vs fine-tuned comparisons."""

from __future__ import annotations

import math

import numpy as np


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    if not top_k:
        return 0.0
    return float(sum(1 for doc_id in top_k if doc_id in relevant_set)) / float(k)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    return float(sum(1 for doc_id in top_k if doc_id in relevant_set)) / float(len(relevant_set))


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    gains = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))

    ideal_hits = min(k, len(relevant_set))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    score_sum = 0.0
    hits = 0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            hits += 1
            score_sum += hits / float(rank)

    return score_sum / float(len(relevant_set))


def summarize_retrieval_metrics(results: list[dict]) -> dict[str, float]:
    """Aggregate retrieval metrics over query-level ranked lists."""
    recall_1, recall_5, recall_10 = [], [], []
    precision_1, precision_5, precision_10 = [], [], []
    mrr_scores, ndcg_scores, map_scores = [], [], []

    for row in results:
        retrieved = row["retrieved_ids"]
        relevant = row["relevant_ids"]

        recall_1.append(recall_at_k(retrieved, relevant, 1))
        recall_5.append(recall_at_k(retrieved, relevant, 5))
        recall_10.append(recall_at_k(retrieved, relevant, 10))

        precision_1.append(precision_at_k(retrieved, relevant, 1))
        precision_5.append(precision_at_k(retrieved, relevant, 5))
        precision_10.append(precision_at_k(retrieved, relevant, 10))

        mrr_scores.append(reciprocal_rank(retrieved, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, 10))
        map_scores.append(average_precision(retrieved, relevant, 10))

    return {
        "recall@1": float(np.mean(recall_1)) if recall_1 else 0.0,
        "recall@5": float(np.mean(recall_5)) if recall_5 else 0.0,
        "recall@10": float(np.mean(recall_10)) if recall_10 else 0.0,
        "precision@1": float(np.mean(precision_1)) if precision_1 else 0.0,
        "precision@5": float(np.mean(precision_5)) if precision_5 else 0.0,
        "precision@10": float(np.mean(precision_10)) if precision_10 else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "map@10": float(np.mean(map_scores)) if map_scores else 0.0,
    }
