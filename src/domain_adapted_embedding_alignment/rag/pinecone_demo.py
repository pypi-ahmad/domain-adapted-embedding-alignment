"""Pinecone integration for live retrieval benchmarking."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from domain_adapted_embedding_alignment.retrieval.embeddings import EmbeddingBackend


def _pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Set it in .env or shell environment.")
    return Pinecone(api_key=api_key)


def _index_names(pc: Pinecone) -> set[str]:
    listing = pc.list_indexes()
    if hasattr(listing, "names"):
        try:
            return set(listing.names())
        except Exception:  # noqa: BLE001
            pass

    names: set[str] = set()
    try:
        for entry in listing:
            if isinstance(entry, dict) and "name" in entry:
                names.add(str(entry["name"]))
            elif hasattr(entry, "name"):
                names.add(str(entry.name))
            else:
                names.add(str(entry))
    except Exception:  # noqa: BLE001
        return set()
    return names


def build_pinecone_index(
    index_name: str,
    doc_ids: list[str],
    doc_texts: list[str],
    doc_domains: list[str],
    backend: EmbeddingBackend,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Create or reset Pinecone index and upload document embeddings."""
    pc = _pinecone_client()

    vectors = backend.embed_texts(doc_texts, normalize=True)
    dim = int(vectors.shape[1])

    existing = _index_names(pc)
    if index_name in existing:
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    index = pc.Index(index_name)

    upserts: list[dict[str, Any]] = []
    for doc_id, vec, text, domain in zip(doc_ids, vectors, doc_texts, doc_domains, strict=False):
        upserts.append(
            {
                "id": doc_id,
                "values": vec.tolist(),
                "metadata": {"domain": domain, "text": text[:1200]},
            }
        )

    batch_size = 100
    for start in range(0, len(upserts), batch_size):
        index.upsert(vectors=upserts[start : start + batch_size])

    logger.info("Pinecone index built: {} ({} docs)", index_name, len(doc_ids))


def search_pinecone(
    index_name: str,
    query: str,
    backend: EmbeddingBackend,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Run query-time retrieval from Pinecone."""
    pc = _pinecone_client()
    index = pc.Index(index_name)

    query_vec = backend.embed_texts([query], normalize=True)[0].tolist()
    response = index.query(vector=query_vec, top_k=top_k, include_metadata=False)

    if isinstance(response, dict):
        matches = response.get("matches", [])
        return [(match["id"], float(match["score"])) for match in matches]

    matches = getattr(response, "matches", [])
    normalized = []
    for match in matches:
        if isinstance(match, dict):
            normalized.append((str(match["id"]), float(match["score"])))
        else:
            normalized.append((str(match.id), float(match.score)))
    return normalized
