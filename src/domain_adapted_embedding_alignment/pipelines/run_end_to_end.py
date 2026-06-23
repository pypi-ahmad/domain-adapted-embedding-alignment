"""Top-level orchestration for full local project execution."""

from __future__ import annotations

from loguru import logger
import polars as pl

from domain_adapted_embedding_alignment.evaluation.visualization import plot_embedding_projections
from domain_adapted_embedding_alignment.pipelines.evaluate_models import run_evaluation
from domain_adapted_embedding_alignment.pipelines.finalize_reports import (
    build_completion_checklist,
    build_tool_impact_report,
)
from domain_adapted_embedding_alignment.pipelines.prepare_data import run_prepare_data
from domain_adapted_embedding_alignment.pipelines.run_graphrag_benchmarks import run_graphrag_benchmarks
from domain_adapted_embedding_alignment.pipelines.run_inference import run_demo_inference
from domain_adapted_embedding_alignment.pipelines.run_rag_benchmarks import run_rag_benchmarks
from domain_adapted_embedding_alignment.pipelines.train_model import run_train
from domain_adapted_embedding_alignment.retrieval.backend_factory import build_baseline_backend, build_tuned_backend
from domain_adapted_embedding_alignment.retrieval.embeddings import HuggingFaceEmbeddingBackend
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import save_json


DEFAULT_INFERENCE_QUERIES = [
    "What treatments are available for a brain tumor?",
    "What are legal grounds for termination of contract?",
    "How does credential theft relate to account compromise techniques?",
]


def run_end_to_end(
    settings: Settings,
    run_judge: bool = True,
    training_backend: str | None = None,
) -> dict:
    """Run complete pipeline from data prep to inference artifacts."""
    docs_path = settings.processed_data_dir / "documents.parquet"
    queries_path = settings.processed_data_dir / "queries.parquet"
    pairs_path = settings.processed_data_dir / "pairs.parquet"

    if settings.skip_data_preparation_if_present and docs_path.exists() and queries_path.exists() and pairs_path.exists():
        logger.info("Skipping data preparation and reusing existing processed artifacts")
        data_report_path = settings.reports_dir / "data_preparation_report.json"
        if data_report_path.exists():
            import json

            data_report = json.loads(data_report_path.read_text(encoding="utf-8"))
        else:
            data_report = {
                "reused_processed_data": True,
                "paths": {
                    "documents": str(docs_path),
                    "queries": str(queries_path),
                    "pairs": str(pairs_path),
                },
            }
    else:
        data_report = run_prepare_data(settings)
    train_report = run_train(settings, training_backend=training_backend)
    eval_report = run_evaluation(settings, run_judge=run_judge)

    # Embedding projection figures before vs after fine-tuning.
    docs = pl.read_parquet(settings.processed_data_dir / "documents.parquet").to_dicts()
    sample_docs = docs[: settings.eval_doc_limit]
    doc_texts = [row["text"] for row in sample_docs]
    labels = [row["domain"] for row in sample_docs]

    baseline_backend = build_baseline_backend(settings)
    tuned_backend = build_tuned_backend(settings)

    try:
        baseline_emb = baseline_backend.embed_texts(doc_texts, normalize=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Baseline embedding projection failed with baseline backend ({}). Using HF base fallback.",
            exc,
        )
        baseline_backend = HuggingFaceEmbeddingBackend(
            model_name=settings.trainable_model_name,
            adapter_path=None,
            max_length=settings.max_doc_length,
            batch_size=settings.eval_batch_size,
        )
        baseline_emb = baseline_backend.embed_texts(doc_texts, normalize=True)
    tuned_emb = tuned_backend.embed_texts(doc_texts, normalize=True)

    baseline_figs = plot_embedding_projections(
        embeddings=baseline_emb,
        labels=labels,
        output_dir=settings.figures_dir,
        prefix="baseline",
    )
    tuned_figs = plot_embedding_projections(
        embeddings=tuned_emb,
        labels=labels,
        output_dir=settings.figures_dir,
        prefix="tuned",
    )

    rag_report = run_rag_benchmarks(settings)
    graphrag_report = run_graphrag_benchmarks(settings)
    inference_examples = run_demo_inference(settings, queries=DEFAULT_INFERENCE_QUERIES)

    final_report = {
        "data_preparation": data_report,
        "training": train_report,
        "evaluation": eval_report,
        "visualizations": {
            "baseline": baseline_figs,
            "tuned": tuned_figs,
            "backend_metadata": {
                "baseline_backend": baseline_backend.__class__.__name__,
                "tuned_backend": tuned_backend.__class__.__name__,
            },
        },
        "rag": rag_report,
        "graphrag": graphrag_report,
        "inference_examples": inference_examples,
    }

    save_json(final_report, settings.reports_dir / "final_report.json")
    completion = build_completion_checklist(settings)
    tool_impact = build_tool_impact_report(settings)
    final_report["completion"] = completion
    final_report["tool_impact"] = tool_impact
    save_json(final_report, settings.reports_dir / "final_report.json")
    logger.info("Full end-to-end pipeline completed")
    return final_report
