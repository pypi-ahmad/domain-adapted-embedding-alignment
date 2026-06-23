"""Typed data contracts for dataset preparation, training, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    """One retrieval document with domain metadata."""

    doc_id: str
    domain: str
    source: str
    text: str
    title: str = ""
    label: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = payload["metadata"] or {}
        return payload


@dataclass(slots=True)
class PairRecord:
    """Training/evaluation pair including explicit hard negatives."""

    pair_id: str
    split: str
    domain: str
    source: str
    query: str
    positive_doc_id: str
    positive_text: str
    negative_doc_id: str
    negative_text: str
    hard_negative_doc_id: str
    hard_negative_text: str
    relevance: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    """A ranked retrieval hit for one query."""

    query_id: str
    query: str
    doc_id: str
    score: float
    rank: int
    domain: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalQuery:
    """Held-out query with one or more relevant document IDs."""

    query_id: str
    domain: str
    query: str
    relevant_doc_ids: list[str]
    reference_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
