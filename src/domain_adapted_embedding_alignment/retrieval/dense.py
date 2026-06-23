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
        query_embedding = self.backend.embed_texts([query], normalize=True)[0]
        scores = self.doc_embeddings @ query_embedding
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[int(index)], float(scores[int(index)])) for index in ranked]
