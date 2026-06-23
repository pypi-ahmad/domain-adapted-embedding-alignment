"""ChromaDB integration for baseline vs fine-tuned retrieval demos."""

from __future__ import annotations

from pathlib import Path

import chromadb
from loguru import logger

from domain_adapted_embedding_alignment.retrieval.embeddings import EmbeddingBackend


def build_chroma_index(
    chroma_dir: Path,
    collection_name: str,
    doc_ids: list[str],
    doc_texts: list[str],
    doc_domains: list[str],
    backend: EmbeddingBackend,
) -> None:
    """Build a Chroma collection from local corpus embeddings."""
    client = chromadb.PersistentClient(path=str(chroma_dir))

    existing = {item.name for item in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    vectors = backend.embed_texts(doc_texts, normalize=True).tolist()
    metadatas = [{"domain": domain} for domain in doc_domains]

    collection.add(
        ids=doc_ids,
        embeddings=vectors,
        documents=doc_texts,
        metadatas=metadatas,
    )
    logger.info("Chroma index built: {} ({} docs)", collection_name, len(doc_ids))


def search_chroma(
    chroma_dir: Path,
    collection_name: str,
    query: str,
    backend: EmbeddingBackend,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Search Chroma with external query embeddings."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)

    query_vec = backend.embed_texts([query], normalize=True)[0].tolist()
    response = collection.query(query_embeddings=[query_vec], n_results=top_k)

    ids = response.get("ids", [[]])[0]
    distances = response.get("distances", [[]])[0]

    # Convert cosine distance to similarity-like score.
    scores = [1.0 - float(dist) for dist in distances]
    return list(zip(ids, scores))
