"""Assemble final report from completed pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

from domain_adapted_embedding_alignment.pipelines.finalize_reports import (
    build_completion_checklist,
    build_tool_impact_report,
)
from domain_adapted_embedding_alignment.pipelines.run_end_to_end import DEFAULT_INFERENCE_QUERIES
from domain_adapted_embedding_alignment.pipelines.run_inference import run_demo_inference
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import load_json, save_json


def _load_required(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def run_build_final_report(settings: Settings, queries: list[str] | None = None) -> dict:
    """Build `final_report.json` from existing stage outputs plus fresh inference."""
    data_report = _load_required(settings.reports_dir / "data_preparation_report.json")
    train_report = _load_required(settings.reports_dir / "training_report.json")
    eval_report = _load_required(settings.eval_dir / "retrieval_evaluation.json")
    rag_report = _load_required(settings.eval_dir / "rag_benchmark.json")
    graphrag_report = _load_required(settings.eval_dir / "graphrag_benchmark.json")

    baseline = {
        "pca": str(settings.figures_dir / "baseline_pca.png"),
        "tsne": str(settings.figures_dir / "baseline_tsne.png"),
        "umap": str(settings.figures_dir / "baseline_umap.png"),
    }
    tuned = {
        "pca": str(settings.figures_dir / "tuned_pca.png"),
        "tsne": str(settings.figures_dir / "tuned_tsne.png"),
        "umap": str(settings.figures_dir / "tuned_umap.png"),
    }

    used_queries = queries or list(DEFAULT_INFERENCE_QUERIES)
    inference_examples = run_demo_inference(settings, queries=used_queries)

    final_report = {
        "data_preparation": data_report,
        "training": train_report,
        "evaluation": eval_report,
        "visualizations": {
            "baseline": baseline,
            "tuned": tuned,
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
    return final_report
