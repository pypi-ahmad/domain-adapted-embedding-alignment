"""Comprehensive evaluation suite for baseline vs fine-tuned embedding retrieval."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import silhouette_score

from domain_adapted_embedding_alignment.evaluation.llm_judge import judge_retrieval_context
from domain_adapted_embedding_alignment.retrieval.metrics import summarize_retrieval_metrics
from domain_adapted_embedding_alignment.schemas import EvalQuery
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import latency_summary


def _context_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in retrieved_ids if item in relevant_set)
    return hits / float(len(retrieved_ids))


def _context_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in retrieved_ids if item in relevant_set)
    return hits / float(len(relevant_set))


def _ranking_accuracy(sim_pos: list[float], sim_neg: list[float]) -> float:
    if not sim_pos:
        return 0.0
    comparisons = [1.0 if p > n else 0.0 for p, n in zip(sim_pos, sim_neg, strict=False)]
    return float(np.mean(comparisons)) if comparisons else 0.0


def _cluster_metrics(embeddings: np.ndarray, labels: list[str]) -> dict[str, float]:
    if len(embeddings) < 10:
        return {
            "silhouette_score": 0.0,
            "intra_cluster_distance": 0.0,
            "inter_cluster_distance": 0.0,
        }

    try:
        sil = float(silhouette_score(embeddings, labels, metric="cosine"))
    except Exception:
        sil = 0.0

    # Intra/inter distance summary.
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    intra_values: list[float] = []
    inter_values: list[float] = []

    for _label, idxs in label_to_indices.items():
        if len(idxs) < 2:
            continue
        subset = embeddings[idxs]
        for i in range(len(subset) - 1):
            for j in range(i + 1, len(subset)):
                intra_values.append(float(np.linalg.norm(subset[i] - subset[j])))

    all_labels = list(label_to_indices)
    for i in range(len(all_labels) - 1):
        for j in range(i + 1, len(all_labels)):
            a = embeddings[label_to_indices[all_labels[i]]]
            b = embeddings[label_to_indices[all_labels[j]]]
            # Mean pairwise centroid distance approximation.
            inter_values.append(float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))))

    return {
        "silhouette_score": sil,
        "intra_cluster_distance": float(np.mean(intra_values)) if intra_values else 0.0,
        "inter_cluster_distance": float(np.mean(inter_values)) if inter_values else 0.0,
    }


def evaluate_retrieval_system(
    queries: list[EvalQuery],
    retriever_callable,
    doc_lookup: dict[str, dict],
    settings: Settings,
    run_llm_judge: bool = True,
) -> dict:
    """Evaluate one retrieval system against held-out query relevance."""

    query_rows: list[dict] = []
    judge_candidates: list[dict[str, str]] = []
    retrieval_latencies: list[float] = []

    for query in queries:
        start = time.perf_counter()
        hits = retriever_callable(query.query, top_k=settings.retrieval_top_k)
        retrieval_latencies.append(time.perf_counter() - start)
        retrieved_ids = [doc_id for doc_id, _ in hits]

        row = {
            "query_id": query.query_id,
            "domain": query.domain,
            "query": query.query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": query.relevant_doc_ids,
        }
        query_rows.append(row)

        if run_llm_judge and len(judge_candidates) < settings.judge_max_queries:
            context_docs = [doc_lookup[doc_id]["text"] for doc_id in retrieved_ids if doc_id in doc_lookup][:4]
            context = "\n\n".join(context_docs)
            retrieved_summary = " | ".join(
                [doc_lookup[doc_id].get("label", "") for doc_id in retrieved_ids if doc_id in doc_lookup][:4]
            )

            judge_candidates.append(
                {
                    "query_id": query.query_id,
                    "domain": query.domain,
                    "query": query.query,
                    "context": context,
                    "retrieved_summary": retrieved_summary,
                }
            )

    judge_rows: list[dict] = []
    if judge_candidates:
        max_workers = min(8, len(judge_candidates))

        def _score_candidate(candidate: dict[str, str]) -> dict:
            judge_primary = judge_retrieval_context(
                settings.judge_model_primary,
                candidate["query"],
                candidate["context"],
                candidate["retrieved_summary"],
            )
            judge_secondary = judge_retrieval_context(
                settings.judge_model_secondary,
                candidate["query"],
                candidate["context"],
                candidate["retrieved_summary"],
            )
            return {
                "query_id": candidate["query_id"],
                "domain": candidate["domain"],
                "primary_relevance": judge_primary.relevance,
                "primary_usefulness": judge_primary.usefulness,
                "primary_semantic_relevance": judge_primary.semantic_relevance,
                "secondary_relevance": judge_secondary.relevance,
                "secondary_usefulness": judge_secondary.usefulness,
                "secondary_semantic_relevance": judge_secondary.semantic_relevance,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            judge_rows = list(executor.map(_score_candidate, judge_candidates))

    retrieval_metrics = summarize_retrieval_metrics(query_rows)

    context_precision_scores = [
        _context_precision(row["retrieved_ids"], row["relevant_ids"])
        for row in query_rows
    ]
    context_recall_scores = [
        _context_recall(row["retrieved_ids"], row["relevant_ids"])
        for row in query_rows
    ]

    rag_metrics = {
        "context_precision": float(np.mean(context_precision_scores)) if context_precision_scores else 0.0,
        "context_recall": float(np.mean(context_recall_scores)) if context_recall_scores else 0.0,
        # Faithfulness proxy: average context precision and judge relevance.
        "retrieval_faithfulness_proxy": float(np.mean(context_precision_scores)) if context_precision_scores else 0.0,
    }

    judge_metrics = {}
    if judge_rows:
        relevance = [
            (row["primary_relevance"] + row["secondary_relevance"]) / 10.0
            for row in judge_rows
        ]
        usefulness = [
            (row["primary_usefulness"] + row["secondary_usefulness"]) / 10.0
            for row in judge_rows
        ]
        semantic = [
            (row["primary_semantic_relevance"] + row["secondary_semantic_relevance"]) / 10.0
            for row in judge_rows
        ]
        judge_metrics = {
            "judge_relevance": float(np.mean(relevance)),
            "judge_usefulness": float(np.mean(usefulness)),
            "judge_semantic_relevance": float(np.mean(semantic)),
        }

    # Domain-level retrieval summaries.
    domain_rows: dict[str, list[dict]] = defaultdict(list)
    for row in query_rows:
        domain_rows[row["domain"]].append(row)

    by_domain = {
        domain: summarize_retrieval_metrics(rows)
        for domain, rows in domain_rows.items()
    }

    return {
        "retrieval_metrics": retrieval_metrics,
        "latency_metrics": latency_summary(retrieval_latencies),
        "rag_metrics": rag_metrics,
        "judge_metrics": judge_metrics,
        "by_domain": by_domain,
        "query_level": query_rows,
    }


def evaluate_similarity_and_clusters(
    query_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    hard_negative_embeddings: np.ndarray,
    domain_labels: list[str],
) -> dict:
    """Compute embedding-space metrics."""
    cosine_pos = np.sum(query_embeddings * positive_embeddings, axis=1)
    cosine_neg = np.sum(query_embeddings * hard_negative_embeddings, axis=1)

    ranking_accuracy = _ranking_accuracy(cosine_pos.tolist(), cosine_neg.tolist())

    # Build one matrix for cluster metrics.
    combined = np.concatenate([query_embeddings, positive_embeddings], axis=0)
    combined_labels = domain_labels + domain_labels
    cluster_metrics = _cluster_metrics(combined, combined_labels)

    return {
        "cosine_similarity_positive": float(np.mean(cosine_pos)) if len(cosine_pos) else 0.0,
        "cosine_similarity_negative": float(np.mean(cosine_neg)) if len(cosine_neg) else 0.0,
        "ranking_accuracy": ranking_accuracy,
        **cluster_metrics,
    }


def save_evaluation_report(payload: dict, path: Path) -> None:
    """Persist evaluation payload to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to {}", path)
