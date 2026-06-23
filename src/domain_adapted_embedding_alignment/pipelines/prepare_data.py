"""End-to-end data preparation pipeline."""

from __future__ import annotations

from dataclasses import asdict

from loguru import logger

from domain_adapted_embedding_alignment.data.loaders import (
    load_cyber_mitre,
    load_legal_lexglue,
    load_medical_medmentions,
    load_msmarco,
    locate_mitre_cti_root,
)
from domain_adapted_embedding_alignment.data.pair_builder import (
    build_pairs,
    persist_documents,
    persist_pairs,
    persist_queries,
)
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import save_json


def run_prepare_data(settings: Settings) -> dict:
    """Prepare corpus, query seeds, and training pairs.

    Returns a metadata payload that downstream pipelines can reuse.
    """
    logger.info("Starting data preparation")

    ms_docs, ms_queries = load_msmarco(settings.msmarco_max_rows, seed=settings.random_seed)
    med_docs, med_queries = load_medical_medmentions(settings.medical_max_records, seed=settings.random_seed)
    legal_docs, legal_queries = load_legal_lexglue(settings.legal_max_records, seed=settings.random_seed)

    mitre_root = locate_mitre_cti_root(settings.project_root)
    cyber_docs, cyber_queries = load_cyber_mitre(settings.cyber_max_records, mitre_root=mitre_root)

    documents = ms_docs + med_docs + legal_docs + cyber_docs
    queries = ms_queries + med_queries + legal_queries + cyber_queries

    pairs = build_pairs(
        documents=documents,
        queries=queries,
        target_size=settings.final_pair_target,
        seed=settings.random_seed,
    )

    docs_path = settings.processed_data_dir / "documents.parquet"
    queries_path = settings.processed_data_dir / "queries.parquet"
    pairs_path = settings.processed_data_dir / "pairs.parquet"

    persist_documents(documents, str(docs_path))
    persist_queries(queries, str(queries_path))
    persist_pairs(pairs, str(pairs_path))

    stats = {
        "documents_total": len(documents),
        "queries_total": len(queries),
        "pairs_total": len(pairs),
        "documents_by_domain": {
            "general": len(ms_docs),
            "medical": len(med_docs),
            "legal": len(legal_docs),
            "cybersecurity": len(cyber_docs),
        },
        "queries_by_domain": {
            "general": len(ms_queries),
            "medical": len(med_queries),
            "legal": len(legal_queries),
            "cybersecurity": len(cyber_queries),
        },
        "paths": {
            "documents": str(docs_path),
            "queries": str(queries_path),
            "pairs": str(pairs_path),
        },
    }

    save_json(stats, settings.reports_dir / "data_preparation_report.json")
    save_json([asdict(item) for item in queries[:5000]], settings.eval_dir / "eval_queries_sample.json")

    logger.info("Data preparation completed")
    return stats
