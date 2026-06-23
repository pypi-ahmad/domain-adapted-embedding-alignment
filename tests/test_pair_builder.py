"""Tests for split stability and pair construction sanity."""

from domain_adapted_embedding_alignment.data.pair_builder import _split_from_query_id


def test_split_assignment_is_stable() -> None:
    qid = "example_query_123"
    assert _split_from_query_id(qid) == _split_from_query_id(qid)


def test_split_assignment_domain() -> None:
    split = _split_from_query_id("another_query")
    assert split in {"train", "validation", "test"}
