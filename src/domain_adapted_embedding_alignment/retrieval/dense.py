"""Dense retrieval utilities."""

from __future__ import annotations

import numpy as np

from domain_adapted_embedding_alignment.retrieval.embeddings import EmbeddingBackend


class DenseRetriever:
    """Dense retriever using cosine-similar normalized embeddings."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        doc_ids: list[str],
        doc_texts: list[str],
        batch_size: int = 16,
    ) -> None:
        self.backend = backend
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.batch_size = batch_size

        self.doc_embeddings = self.backend.embed_texts(doc_texts, normalize=True)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        query_embedding = self.backend.embed_texts([query], normalize=True)[0]
        scores = self.doc_embeddings @ query_embedding
        k = min(top_k, int(scores.shape[0]))
        if k == 0:
            return []

        if k == int(scores.shape[0]):
            ranked = np.argsort(scores)[::-1]
        else:
            candidate_idx = np.argpartition(scores, -k)[-k:]
            ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

        return [(self.doc_ids[int(index)], float(scores[int(index)])) for index in ranked]
