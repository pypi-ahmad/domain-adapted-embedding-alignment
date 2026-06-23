"""Unit tests for retrieval metric correctness."""

from domain_adapted_embedding_alignment.retrieval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_recall_at_k() -> None:
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d2", "d4"]

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5


def test_reciprocal_rank() -> None:
    retrieved = ["x", "y", "z"]
    relevant = ["z"]
    assert reciprocal_rank(retrieved, relevant) == 1 / 3


def test_ndcg_non_negative() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = ["d"]
    assert 0.0 <= ndcg_at_k(retrieved, relevant, 4) <= 1.0
