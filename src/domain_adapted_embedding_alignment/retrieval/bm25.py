"""Sparse BM25 retrieval baseline."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())


class BM25Retriever:
    """Simple BM25 retriever over a static in-memory corpus."""

    def __init__(self, doc_ids: list[str], doc_texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self._tokenized_docs = [_tokenize(text) for text in doc_texts]
        self._bm25 = BM25Okapi(self._tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, int(scores.shape[0]))
        if k == 0:
            return []

        if k == int(scores.shape[0]):
            idx = np.argsort(scores)[::-1]
        else:
            candidate_idx = np.argpartition(scores, -k)[-k:]
            idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]
