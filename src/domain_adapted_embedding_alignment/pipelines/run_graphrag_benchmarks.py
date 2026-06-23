"""GraphRAG benchmark pipeline for local/global search comparisons."""

from __future__ import annotations

import time

import networkx as nx
import polars as pl
import pickle
from loguru import logger

from domain_adapted_embedding_alignment.graphrag.graph_builder import build_document_entity_graph
from domain_adapted_embedding_alignment.graphrag.retriever import (
    global_community_retrieval,
    local_graph_retrieval,
)
from domain_adapted_embedding_alignment.retrieval.backend_factory import build_baseline_backend, build_tuned_backend
from domain_adapted_embedding_alignment.retrieval.dense import DenseRetriever
from domain_adapted_embedding_alignment.retrieval.embeddings import HuggingFaceEmbeddingBackend
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import latency_summary, save_json


def _hit_at_k(hits: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
    subset = hits[:k]
    return 1.0 if any(doc_id in relevant for doc_id, _ in subset) else 0.0


def run_graphrag_benchmarks(settings: Settings) -> dict:
    docs = (
        pl.read_parquet(settings.processed_data_dir / "documents.parquet")
        .head(settings.graphrag_doc_limit)
        .to_dicts()
    )
    queries = pl.read_parquet(settings.processed_data_dir / "queries.parquet").to_dicts()[
        : settings.graphrag_query_limit
    ]

    doc_ids = [str(row["doc_id"]) for row in docs]
    doc_texts = [str(row["text"]) for row in docs]

    graph, communities = build_document_entity_graph(docs)
    for row in docs:
        graph.nodes[row["doc_id"]]["text"] = row["text"]

    with open(settings.artifacts_dir / "graph_document_entity.gpickle", "wb") as file_handle:
        pickle.dump(graph, file_handle)
    save_json(communities, settings.artifacts_dir / "graph_communities.json")

    baseline_backend = build_baseline_backend(settings)
    try:
        _ = baseline_backend.embed_texts(doc_texts[:8], normalize=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Baseline backend failed for GraphRAG benchmark ({}). Using HF base fallback.",
            exc,
        )
        baseline_backend = HuggingFaceEmbeddingBackend(
            model_name=settings.trainable_model_name,
            adapter_path=None,
            max_length=settings.max_doc_length,
            batch_size=settings.eval_batch_size,
        )
    tuned_backend = build_tuned_backend(settings)

    baseline_dense = DenseRetriever(baseline_backend, doc_ids, doc_texts)
    tuned_dense = DenseRetriever(tuned_backend, doc_ids, doc_texts)

    baseline_rows = []
    tuned_rows = []
    baseline_dense_latencies: list[float] = []
    tuned_dense_latencies: list[float] = []
    baseline_local_latencies: list[float] = []
    tuned_local_latencies: list[float] = []
    baseline_global_latencies: list[float] = []
    tuned_global_latencies: list[float] = []

    for query in queries:
        q_text = str(query["query"])
        relevant = set(query["relevant_doc_ids"])

        start = time.perf_counter()
        b_dense = baseline_dense.search(q_text, top_k=10)
        baseline_dense_latencies.append(time.perf_counter() - start)
        start = time.perf_counter()
        t_dense = tuned_dense.search(q_text, top_k=10)
        tuned_dense_latencies.append(time.perf_counter() - start)

        start = time.perf_counter()
        b_local = local_graph_retrieval(graph, b_dense, hops=settings.graph_neighbor_hops, top_k=10)
        baseline_local_latencies.append(time.perf_counter() - start)
        start = time.perf_counter()
        t_local = local_graph_retrieval(graph, t_dense, hops=settings.graph_neighbor_hops, top_k=10)
        tuned_local_latencies.append(time.perf_counter() - start)

        start = time.perf_counter()
        b_global = global_community_retrieval(graph, communities, q_text, top_k=10)
        baseline_global_latencies.append(time.perf_counter() - start)
        start = time.perf_counter()
        t_global = global_community_retrieval(graph, communities, q_text, top_k=10)
        tuned_global_latencies.append(time.perf_counter() - start)

        baseline_rows.append(
            {
                "local_hit@5": _hit_at_k(b_local, relevant, 5),
                "global_hit@5": _hit_at_k(b_global, relevant, 5),
            }
        )
        tuned_rows.append(
            {
                "local_hit@5": _hit_at_k(t_local, relevant, 5),
                "global_hit@5": _hit_at_k(t_global, relevant, 5),
            }
        )

    payload = {
        "baseline": {
            "local_hit_rate@5": float(sum(row["local_hit@5"] for row in baseline_rows) / max(1, len(baseline_rows))),
            "global_hit_rate@5": float(sum(row["global_hit@5"] for row in baseline_rows) / max(1, len(baseline_rows))),
            "dense_latency": latency_summary(baseline_dense_latencies),
            "local_latency": latency_summary(baseline_local_latencies),
            "global_latency": latency_summary(baseline_global_latencies),
        },
        "tuned": {
            "local_hit_rate@5": float(sum(row["local_hit@5"] for row in tuned_rows) / max(1, len(tuned_rows))),
            "global_hit_rate@5": float(sum(row["global_hit@5"] for row in tuned_rows) / max(1, len(tuned_rows))),
            "dense_latency": latency_summary(tuned_dense_latencies),
            "local_latency": latency_summary(tuned_local_latencies),
            "global_latency": latency_summary(tuned_global_latencies),
        },
        "n_queries": len(queries),
    }

    save_json(payload, settings.eval_dir / "graphrag_benchmark.json")
    return payload
