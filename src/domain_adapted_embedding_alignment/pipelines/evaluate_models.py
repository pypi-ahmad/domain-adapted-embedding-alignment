"""Baseline and fine-tuned retrieval evaluation pipeline."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from loguru import logger

from domain_adapted_embedding_alignment.evaluation.evaluator import (
    evaluate_retrieval_system,
    evaluate_similarity_and_clusters,
    save_evaluation_report,
)
from domain_adapted_embedding_alignment.retrieval.backend_factory import build_baseline_backend, build_tuned_backend
from domain_adapted_embedding_alignment.retrieval.bm25 import BM25Retriever
from domain_adapted_embedding_alignment.retrieval.dense import DenseRetriever
from domain_adapted_embedding_alignment.retrieval.embeddings import HuggingFaceEmbeddingBackend
from domain_adapted_embedding_alignment.retrieval.hybrid import reciprocal_rank_fusion, weighted_score_fusion
from domain_adapted_embedding_alignment.schemas import EvalQuery
from domain_adapted_embedding_alignment.settings import Settings


def _load_inputs(settings: Settings) -> tuple[list[dict], list[EvalQuery], list[dict]]:
    docs_frame = pl.read_parquet(settings.processed_data_dir / "documents.parquet").head(settings.eval_doc_limit)
    queries_frame = pl.read_parquet(settings.processed_data_dir / "queries.parquet")
    pairs_frame = pl.read_parquet(settings.processed_data_dir / "pairs.parquet")

    docs = docs_frame.to_dicts()
    eval_queries = [EvalQuery(**row) for row in queries_frame.filter(pl.col("query_id").is_not_null()).to_dicts()]
    eval_queries = eval_queries[: settings.eval_query_limit]
    pairs = pairs_frame.filter(pl.col("split") == "test").to_dicts()[: settings.similarity_sample_limit]
    return docs, eval_queries, pairs


def run_evaluation(settings: Settings, run_judge: bool = True) -> dict:
    """Evaluate BM25, dense baseline, dense tuned, and hybrid retrieval."""
    docs, eval_queries, test_pairs = _load_inputs(settings)

    doc_ids = [str(row["doc_id"]) for row in docs]
    doc_texts = [str(row["text"]) for row in docs]
    doc_domains = [str(row["domain"]) for row in docs]
    doc_lookup = {str(row["doc_id"]): row for row in docs}

    logger.info("Building retrieval systems for evaluation")
    bm25 = BM25Retriever(doc_ids=doc_ids, doc_texts=doc_texts)

    baseline_backend = build_baseline_backend(settings)
    baseline_backend_name = settings.baseline_embedding_model
    try:
        baseline_dense = DenseRetriever(
            backend=baseline_backend,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Baseline backend '{}' failed during corpus indexing ({}). Falling back to HF base model.",
            settings.baseline_embedding_model,
            exc,
        )
        baseline_backend = HuggingFaceEmbeddingBackend(
            model_name=settings.trainable_model_name,
            adapter_path=None,
            max_length=settings.max_doc_length,
            batch_size=settings.eval_batch_size,
        )
        baseline_backend_name = f"{settings.trainable_model_name} (fallback)"
        baseline_dense = DenseRetriever(
            backend=baseline_backend,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
        )

    tuned_backend = build_tuned_backend(settings)
    tuned_dense = DenseRetriever(
        backend=tuned_backend,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
    )

    def bm25_search(query: str, top_k: int = 10):
        return bm25.search(query, top_k=top_k)

    def baseline_search(query: str, top_k: int = 10):
        return baseline_dense.search(query, top_k=top_k)

    def tuned_search(query: str, top_k: int = 10):
        return tuned_dense.search(query, top_k=top_k)

    def hybrid_baseline(query: str, top_k: int = 10):
        dense_hits = baseline_dense.search(query, top_k=top_k * 2)
        sparse_hits = bm25.search(query, top_k=top_k * 2)
        weighted = weighted_score_fusion(dense_hits, sparse_hits, top_k=top_k)
        rrf = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
        return [(doc_id, (w_score + dict(rrf).get(doc_id, 0.0)) / 2.0) for doc_id, w_score in weighted]

    def hybrid_tuned(query: str, top_k: int = 10):
        dense_hits = tuned_dense.search(query, top_k=top_k * 2)
        sparse_hits = bm25.search(query, top_k=top_k * 2)
        weighted = weighted_score_fusion(dense_hits, sparse_hits, top_k=top_k)
        rrf = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
        return [(doc_id, (w_score + dict(rrf).get(doc_id, 0.0)) / 2.0) for doc_id, w_score in weighted]

    bm25_eval = evaluate_retrieval_system(eval_queries, bm25_search, doc_lookup, settings, run_llm_judge=False)
    baseline_eval = evaluate_retrieval_system(eval_queries, baseline_search, doc_lookup, settings, run_llm_judge=run_judge)
    tuned_eval = evaluate_retrieval_system(eval_queries, tuned_search, doc_lookup, settings, run_llm_judge=run_judge)
    hybrid_baseline_eval = evaluate_retrieval_system(
        eval_queries,
        hybrid_baseline,
        doc_lookup,
        settings,
        run_llm_judge=False,
    )
    hybrid_tuned_eval = evaluate_retrieval_system(
        eval_queries,
        hybrid_tuned,
        doc_lookup,
        settings,
        run_llm_judge=False,
    )

    # Similarity and ranking-accuracy diagnostics.
    sample_queries = [row["query"] for row in test_pairs]
    sample_positives = [row["positive_text"] for row in test_pairs]
    sample_negatives = [row["hard_negative_text"] for row in test_pairs]
    sample_domains = [row["domain"] for row in test_pairs]

    baseline_q = baseline_backend.embed_texts(sample_queries, normalize=True)
    baseline_p = baseline_backend.embed_texts(sample_positives, normalize=True)
    baseline_n = baseline_backend.embed_texts(sample_negatives, normalize=True)

    tuned_q = tuned_backend.embed_texts(sample_queries, normalize=True)
    tuned_p = tuned_backend.embed_texts(sample_positives, normalize=True)
    tuned_n = tuned_backend.embed_texts(sample_negatives, normalize=True)

    baseline_similarity = evaluate_similarity_and_clusters(baseline_q, baseline_p, baseline_n, sample_domains)
    tuned_similarity = evaluate_similarity_and_clusters(tuned_q, tuned_p, tuned_n, sample_domains)

    payload = {
        "backend_metadata": {
            "baseline_backend_used": baseline_backend_name,
            "tuned_backend_used": tuned_backend.__class__.__name__,
        },
        "systems": {
            "bm25": bm25_eval,
            "baseline_dense_qwen4b": baseline_eval,
            "tuned_dense_qwen0_6b_lora": tuned_eval,
            "hybrid_baseline": hybrid_baseline_eval,
            "hybrid_tuned": hybrid_tuned_eval,
        },
        "similarity_metrics": {
            "baseline": baseline_similarity,
            "tuned": tuned_similarity,
        },
    }

    save_evaluation_report(payload, settings.eval_dir / "retrieval_evaluation.json")

    # Create compact benchmark table for README/notebooks.
    rows = []
    for system_name, eval_payload in payload["systems"].items():
        metrics = eval_payload["retrieval_metrics"]
        latency = eval_payload.get("latency_metrics", {})
        rows.append(
            {
                "system": system_name,
                "recall@10": metrics.get("recall@10", 0.0),
                "precision@10": metrics.get("precision@10", 0.0),
                "mrr": metrics.get("mrr", 0.0),
                "ndcg@10": metrics.get("ndcg@10", 0.0),
                "map@10": metrics.get("map@10", 0.0),
                "mean_latency_ms": latency.get("mean_ms", 0.0),
                "p95_latency_ms": latency.get("p95_ms", 0.0),
            }
        )
    pl.DataFrame(rows).write_csv(settings.eval_dir / "benchmark_table.csv")

    logger.info("Evaluation completed")
    return payload
