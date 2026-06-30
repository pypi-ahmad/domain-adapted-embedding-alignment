"""Baseline and fine-tuned retrieval evaluation pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from domain_adapted_embedding_alignment.evaluation.evaluator import (
    evaluate_retrieval_system,
    evaluate_similarity_and_clusters,
    save_evaluation_report,
)
from domain_adapted_embedding_alignment.retrieval.backend_factory import (
    build_baseline_backend,
    build_tuned_backend,
)
from domain_adapted_embedding_alignment.retrieval.bm25 import BM25Retriever
from domain_adapted_embedding_alignment.retrieval.dense import DenseRetriever
from domain_adapted_embedding_alignment.retrieval.embeddings import HuggingFaceEmbeddingBackend
from domain_adapted_embedding_alignment.retrieval.hybrid import (
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from domain_adapted_embedding_alignment.schemas import EvalQuery
from domain_adapted_embedding_alignment.settings import Settings


def _load_inputs(settings: Settings) -> tuple[list[dict], list[EvalQuery], list[dict]]:
    docs = (
        pl.scan_parquet(settings.processed_data_dir / "documents.parquet")
        .select(["doc_id", "text", "domain", "source", "label"])
        .limit(settings.eval_doc_limit)
        .collect(streaming=True)
        .to_dicts()
    )
    eval_queries_rows = (
        pl.scan_parquet(settings.processed_data_dir / "queries.parquet")
        .filter(pl.col("query_id").is_not_null())
        .select(["query_id", "domain", "query", "relevant_doc_ids", "reference_answer"])
        .limit(settings.eval_query_limit)
        .collect(streaming=True)
        .to_dicts()
    )
    pairs = (
        pl.scan_parquet(settings.processed_data_dir / "pairs.parquet")
        .filter(pl.col("split") == "test")
        .select(["query", "positive_text", "hard_negative_text", "domain"])
        .limit(settings.similarity_sample_limit)
        .collect(streaming=True)
        .to_dicts()
    )
    eval_queries = [EvalQuery(**row) for row in eval_queries_rows]
    return docs, eval_queries, pairs


def run_evaluation(settings: Settings, run_judge: bool = True) -> dict:
    """Evaluate BM25, dense baseline, dense tuned, and hybrid retrieval."""
    docs, eval_queries, test_pairs = _load_inputs(settings)

    doc_ids = [str(row["doc_id"]) for row in docs]
    doc_texts = [str(row["text"]) for row in docs]
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
        rrf_lookup = dict(rrf)
        return [(doc_id, (w_score + rrf_lookup.get(doc_id, 0.0)) / 2.0) for doc_id, w_score in weighted]

    def hybrid_tuned(query: str, top_k: int = 10):
        dense_hits = tuned_dense.search(query, top_k=top_k * 2)
        sparse_hits = bm25.search(query, top_k=top_k * 2)
        weighted = weighted_score_fusion(dense_hits, sparse_hits, top_k=top_k)
        rrf = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
        rrf_lookup = dict(rrf)
        return [(doc_id, (w_score + rrf_lookup.get(doc_id, 0.0)) / 2.0) for doc_id, w_score in weighted]

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

    if sample_queries:
        n_samples = len(sample_queries)
        stacked_inputs = sample_queries + sample_positives + sample_negatives

        baseline_all = baseline_backend.embed_texts(stacked_inputs, normalize=True)
        baseline_q, baseline_p, baseline_n = np.split(baseline_all, [n_samples, 2 * n_samples])

        tuned_all = tuned_backend.embed_texts(stacked_inputs, normalize=True)
        tuned_q, tuned_p, tuned_n = np.split(tuned_all, [n_samples, 2 * n_samples])

        baseline_similarity = evaluate_similarity_and_clusters(baseline_q, baseline_p, baseline_n, sample_domains)
        tuned_similarity = evaluate_similarity_and_clusters(tuned_q, tuned_p, tuned_n, sample_domains)
    else:
        baseline_similarity = {}
        tuned_similarity = {}

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
