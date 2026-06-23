"""Inference helpers for domain-aware retrieval output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InferenceResult:
    query: str
    ranked_results: list[dict]


def run_inference(query: str, retriever_callable, doc_lookup: dict[str, dict], top_k: int = 10) -> InferenceResult:
    """Retrieve and format ranked outputs for one natural-language query."""
    hits = retriever_callable(query, top_k=top_k)
    ranked = []
    for rank, (doc_id, score) in enumerate(hits, start=1):
        doc = doc_lookup.get(doc_id, {})
        ranked.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": float(score),
                "domain": doc.get("domain", "unknown"),
                "source": doc.get("source", "unknown"),
                "preview": str(doc.get("text", ""))[:220],
            }
        )

    return InferenceResult(query=query, ranked_results=ranked)
