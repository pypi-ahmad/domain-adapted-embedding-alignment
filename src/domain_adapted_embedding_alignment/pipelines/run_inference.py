"""Inference pipeline for domain-specific semantic retrieval."""

from __future__ import annotations

import time

import polars as pl

from domain_adapted_embedding_alignment.rag.inference import run_inference
from domain_adapted_embedding_alignment.retrieval.backend_factory import build_tuned_backend
from domain_adapted_embedding_alignment.retrieval.dense import DenseRetriever
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import latency_summary


def run_demo_inference(settings: Settings, queries: list[str]) -> dict:
    """Run natural-language retrieval examples across target domains."""
    docs = (
        pl.scan_parquet(settings.processed_data_dir / "documents.parquet")
        .select(["doc_id", "text", "domain", "source"])
        .limit(settings.inference_doc_limit)
        .collect(streaming=True)
        .to_dicts()
    )
    doc_ids = [str(row["doc_id"]) for row in docs]
    doc_texts = [str(row["text"]) for row in docs]
    doc_lookup = {str(row["doc_id"]): row for row in docs}

    backend = build_tuned_backend(settings)
    retriever = DenseRetriever(backend=backend, doc_ids=doc_ids, doc_texts=doc_texts)

    results = []
    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        output = run_inference(query, retriever.search, doc_lookup, top_k=settings.retrieval_top_k)
        latencies.append(time.perf_counter() - start)
        results.append({"query": output.query, "ranked_results": output.ranked_results})

    return {
        "results": results,
        "latency": latency_summary(latencies),
        "backend_used": backend.__class__.__name__,
    }
