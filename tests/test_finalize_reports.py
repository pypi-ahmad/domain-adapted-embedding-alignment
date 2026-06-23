"""Tests for completion and tool impact report generation."""

from pathlib import Path

import pytest

from domain_adapted_embedding_alignment.pipelines.finalize_reports import (
    build_completion_checklist,
    build_tool_impact_report,
)
from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import save_json


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "artifacts" / "reports",
        eval_dir=tmp_path / "artifacts" / "evaluation",
        figures_dir=tmp_path / "artifacts" / "figures",
        model_dir=tmp_path / "artifacts" / "models",
        chroma_dir=tmp_path / "chroma_db",
    )
    settings.ensure_directories()
    return settings


def test_completion_checklist_marks_project_complete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    save_json({"status": "ok"}, settings.reports_dir / "data_preparation_report.json")
    save_json(
        {
            "runtime": {
                "backend_requested": "auto",
                "backend_used": "peft",
                "fallback_reason": None,
            }
        },
        settings.reports_dir / "training_report.json",
    )
    save_json({"systems": {}}, settings.eval_dir / "retrieval_evaluation.json")
    save_json({"chroma": {}}, settings.eval_dir / "rag_benchmark.json")
    save_json({"baseline": {}}, settings.eval_dir / "graphrag_benchmark.json")
    save_json({"summary": "ok"}, settings.reports_dir / "final_report.json")
    (settings.figures_dir / "baseline_pca.png").write_bytes(b"x")
    (settings.figures_dir / "tuned_pca.png").write_bytes(b"x")

    payload = build_completion_checklist(settings)
    assert payload["project_completion_ok"] is True


def test_tool_impact_report_computes_deltas(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    save_json(
        {
            "runtime": {
                "backend_requested": "auto",
                "backend_used": "peft",
                "fallback_reason": "CUDA is not available",
            }
        },
        settings.reports_dir / "training_report.json",
    )
    save_json(
        {
            "systems": {
                "baseline_dense_qwen4b": {
                    "retrieval_metrics": {"recall@10": 0.4, "mrr": 0.3, "ndcg@10": 0.2, "map@10": 0.1},
                    "latency_metrics": {"mean_ms": 100.0, "p50_ms": 90.0, "p95_ms": 140.0, "max_ms": 200.0},
                },
                "tuned_dense_qwen0_6b_lora": {
                    "retrieval_metrics": {"recall@10": 0.5, "mrr": 0.35, "ndcg@10": 0.25, "map@10": 0.12},
                    "latency_metrics": {"mean_ms": 120.0, "p50_ms": 100.0, "p95_ms": 160.0, "max_ms": 230.0},
                },
            }
        },
        settings.eval_dir / "retrieval_evaluation.json",
    )
    save_json({"chroma": {"baseline_hit_rate@5": 0.5}}, settings.eval_dir / "rag_benchmark.json")
    save_json({"baseline": {"local_hit_rate@5": 0.5}}, settings.eval_dir / "graphrag_benchmark.json")

    payload = build_tool_impact_report(settings)
    assert payload["peft"]["used"] is True
    assert payload["peft"]["retrieval_delta"]["recall@10"] == pytest.approx(0.1)
    assert payload["peft"]["latency_delta_ms"]["mean_ms"] == pytest.approx(20.0)
    assert payload["unsloth"]["used"] is False
    assert payload["trl"]["used_in_runtime"] is False
