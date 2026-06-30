"""RAG integration benchmarks for ChromaDB and Pinecone."""

from __future__ import annotations

import time

import polars as pl
from loguru import logger

from domain_adapted_embedding_alignment.rag.chroma_demo import build_chroma_index, search_chroma
from domain_adapted_embedding_alignment.rag.pinecone_demo import (
    build_pinecone_index,
    search_pinecone,
)
from domain_adapted_embedding_alignment.retrieval.backend_factory import (
    build_baseline_backend,
    build_tuned_backend,
)
from domain_adapted_embedding_alignment.retrieval.embeddings import HuggingFaceEmbeddingBackend
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import latency_summary, save_json


def run_rag_benchmarks(settings: Settings) -> dict:
    """Build Chroma/Pinecone indexes and compare baseline vs tuned retrieval."""
    docs = (
        pl.scan_parquet(settings.processed_data_dir / "documents.parquet")
        .select(["doc_id", "text", "domain"])
        .limit(settings.rag_doc_limit)
        .collect(streaming=True)
        .to_dicts()
    )
    eval_queries = (
        pl.scan_parquet(settings.processed_data_dir / "queries.parquet")
        .select(["query_id", "domain", "query", "relevant_doc_ids"])
        .limit(settings.rag_query_limit)
        .collect(streaming=True)
        .to_dicts()
    )

    doc_ids = [str(row["doc_id"]) for row in docs]
    doc_texts = [str(row["text"]) for row in docs]
    doc_domains = [str(row["domain"]) for row in docs]

    baseline_backend = build_baseline_backend(settings)
    try:
        _ = baseline_backend.embed_texts(doc_texts[:8], normalize=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Baseline backend failed for RAG benchmark ({}). Using HF base fallback.",
            exc,
        )
        baseline_backend = HuggingFaceEmbeddingBackend(
            model_name=settings.trainable_model_name,
            adapter_path=None,
            max_length=settings.max_doc_length,
            batch_size=settings.eval_batch_size,
        )
    tuned_backend = build_tuned_backend(settings)

    build_chroma_index(
        chroma_dir=settings.chroma_dir,
        collection_name="baseline_qwen4b",
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        doc_domains=doc_domains,
        backend=baseline_backend,
    )
    build_chroma_index(
        chroma_dir=settings.chroma_dir,
        collection_name="tuned_qwen0_6b",
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        doc_domains=doc_domains,
        backend=tuned_backend,
    )

    chroma_rows = []
    chroma_baseline_latencies: list[float] = []
    chroma_tuned_latencies: list[float] = []
    for row in eval_queries:
        query = str(row["query"])
        relevant = set(row["relevant_doc_ids"])

        start = time.perf_counter()
        b_hits = search_chroma(settings.chroma_dir, "baseline_qwen4b", query, baseline_backend, top_k=5)
        chroma_baseline_latencies.append(time.perf_counter() - start)
        start = time.perf_counter()
        t_hits = search_chroma(settings.chroma_dir, "tuned_qwen0_6b", query, tuned_backend, top_k=5)
        chroma_tuned_latencies.append(time.perf_counter() - start)

        b_hit = 1.0 if any(doc_id in relevant for doc_id, _ in b_hits) else 0.0
        t_hit = 1.0 if any(doc_id in relevant for doc_id, _ in t_hits) else 0.0

        chroma_rows.append(
            {
                "query_id": row["query_id"],
                "domain": row["domain"],
                "baseline_hit@5": b_hit,
                "tuned_hit@5": t_hit,
            }
        )

    chroma_summary = {
        "baseline_hit_rate@5": float(sum(r["baseline_hit@5"] for r in chroma_rows) / max(1, len(chroma_rows))),
        "tuned_hit_rate@5": float(sum(r["tuned_hit@5"] for r in chroma_rows) / max(1, len(chroma_rows))),
        "n_queries": len(chroma_rows),
        "baseline_latency": latency_summary(chroma_baseline_latencies),
        "tuned_latency": latency_summary(chroma_tuned_latencies),
    }

    pinecone_summary = {"executed": False, "reason": "disabled"}
    if settings.use_live_pinecone:
        try:
            build_pinecone_index(
                index_name="dea-baseline-qwen4b",
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                doc_domains=doc_domains,
                backend=baseline_backend,
            )
            build_pinecone_index(
                index_name="dea-tuned-qwen06b",
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                doc_domains=doc_domains,
                backend=tuned_backend,
            )

            p_rows = []
            pinecone_baseline_latencies: list[float] = []
            pinecone_tuned_latencies: list[float] = []
            for row in eval_queries:
                query = str(row["query"])
                relevant = set(row["relevant_doc_ids"])
                start = time.perf_counter()
                b_hits = search_pinecone("dea-baseline-qwen4b", query, baseline_backend, top_k=5)
                pinecone_baseline_latencies.append(time.perf_counter() - start)
                start = time.perf_counter()
                t_hits = search_pinecone("dea-tuned-qwen06b", query, tuned_backend, top_k=5)
                pinecone_tuned_latencies.append(time.perf_counter() - start)
                p_rows.append(
                    {
                        "baseline_hit@5": 1.0 if any(doc_id in relevant for doc_id, _ in b_hits) else 0.0,
                        "tuned_hit@5": 1.0 if any(doc_id in relevant for doc_id, _ in t_hits) else 0.0,
                    }
                )

            pinecone_summary = {
                "executed": True,
                "baseline_hit_rate@5": float(sum(r["baseline_hit@5"] for r in p_rows) / max(1, len(p_rows))),
                "tuned_hit_rate@5": float(sum(r["tuned_hit@5"] for r in p_rows) / max(1, len(p_rows))),
                "n_queries": len(p_rows),
                "baseline_latency": latency_summary(pinecone_baseline_latencies),
                "tuned_latency": latency_summary(pinecone_tuned_latencies),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pinecone benchmark skipped: {}", exc)
            pinecone_summary = {"executed": False, "reason": str(exc)}

    payload = {
        "chroma": chroma_summary,
        "pinecone": pinecone_summary,
    }

    save_json(payload, settings.eval_dir / "rag_benchmark.json")
    logger.info("RAG benchmarks completed")
    return payload
