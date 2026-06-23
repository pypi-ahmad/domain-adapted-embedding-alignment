"""Hybrid sparse+dense fusion retrieval."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    top_k: int = 10,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple rankings with reciprocal rank fusion (RRF)."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused[:top_k]


def weighted_score_fusion(
    dense_hits: list[tuple[str, float]],
    sparse_hits: list[tuple[str, float]],
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Fuse dense and sparse score lists after min-max normalization."""
    dense_dict = {doc_id: score for doc_id, score in dense_hits}
    sparse_dict = {doc_id: score for doc_id, score in sparse_hits}
    all_doc_ids = set(dense_dict) | set(sparse_dict)

    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        low, high = min(values), max(values)
        if abs(high - low) < 1e-12:
            return {k: 1.0 for k in scores}
        return {k: (v - low) / (high - low) for k, v in scores.items()}

    dense_norm = _normalize(dense_dict)
    sparse_norm = _normalize(sparse_dict)

    fused: list[tuple[str, float]] = []
    for doc_id in all_doc_ids:
        score = dense_weight * dense_norm.get(doc_id, 0.0) + sparse_weight * sparse_norm.get(doc_id, 0.0)
        fused.append((doc_id, score))

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused[:top_k]
