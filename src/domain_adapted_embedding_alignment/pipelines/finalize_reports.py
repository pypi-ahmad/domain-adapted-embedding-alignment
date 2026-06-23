"""Final report validation and tooling impact summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.utils import load_json, save_json


def _safe_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if isinstance(payload, dict):
        return payload
    return {}


def build_completion_checklist(settings: Settings) -> dict[str, Any]:
    """Validate required end-to-end artifacts and persist a checklist report."""
    required_paths = {
        "data_preparation_report": settings.reports_dir / "data_preparation_report.json",
        "training_report": settings.reports_dir / "training_report.json",
        "retrieval_evaluation": settings.eval_dir / "retrieval_evaluation.json",
        "rag_benchmark": settings.eval_dir / "rag_benchmark.json",
        "graphrag_benchmark": settings.eval_dir / "graphrag_benchmark.json",
        "final_report": settings.reports_dir / "final_report.json",
        "baseline_pca_figure": settings.figures_dir / "baseline_pca.png",
        "tuned_pca_figure": settings.figures_dir / "tuned_pca.png",
    }

    checks: dict[str, dict[str, Any]] = {}
    for name, path in required_paths.items():
        exists = path.exists()
        size_bytes = int(path.stat().st_size) if exists else 0
        checks[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": size_bytes,
            "non_empty": size_bytes > 0,
        }

    training_report = _safe_load(required_paths["training_report"])
    runtime = training_report.get("runtime", {}) if isinstance(training_report, dict) else {}

    runtime_checks = {
        "training_runtime_metadata_present": bool(runtime),
        "backend_requested_present": "backend_requested" in runtime,
        "backend_used_present": "backend_used" in runtime,
        "fallback_reason_present": "fallback_reason" in runtime,
    }

    all_required_ok = all(row["exists"] and row["non_empty"] for row in checks.values())
    all_runtime_ok = all(runtime_checks.values())

    payload = {
        "required_artifacts": checks,
        "runtime_checks": runtime_checks,
        "all_required_artifacts_ok": all_required_ok,
        "all_runtime_checks_ok": all_runtime_ok,
        "project_completion_ok": bool(all_required_ok and all_runtime_ok),
    }
    save_json(payload, settings.reports_dir / "completion_checklist.json")
    return payload


def build_tool_impact_report(settings: Settings) -> dict[str, Any]:
    """Summarize observed PEFT/Unsloth/TRL impact from generated artifacts."""
    training_report = _safe_load(settings.reports_dir / "training_report.json")
    eval_report = _safe_load(settings.eval_dir / "retrieval_evaluation.json")
    rag_report = _safe_load(settings.eval_dir / "rag_benchmark.json")
    graphrag_report = _safe_load(settings.eval_dir / "graphrag_benchmark.json")

    runtime = training_report.get("runtime", {})
    systems = eval_report.get("systems", {})
    baseline = systems.get("baseline_dense_qwen4b", {})
    tuned = systems.get("tuned_dense_qwen0_6b_lora", {})

    b_metrics = baseline.get("retrieval_metrics", {})
    t_metrics = tuned.get("retrieval_metrics", {})
    b_latency = baseline.get("latency_metrics", {})
    t_latency = tuned.get("latency_metrics", {})

    retrieval_delta = {
        metric: float(t_metrics.get(metric, 0.0) - b_metrics.get(metric, 0.0))
        for metric in ["recall@1", "recall@5", "recall@10", "precision@10", "mrr", "ndcg@10", "map@10"]
    }
    latency_delta_ms = {
        metric: float(t_latency.get(metric, 0.0) - b_latency.get(metric, 0.0))
        for metric in ["mean_ms", "p50_ms", "p95_ms", "max_ms"]
    }

    backend_used = runtime.get("backend_used", "unknown")
    requested = runtime.get("backend_requested", "unknown")
    fallback_reason = runtime.get("fallback_reason")

    unsloth_section: dict[str, Any]
    if backend_used == "unsloth":
        unsloth_section = {
            "used": True,
            "status": "executed",
            "note": "Unsloth backend was used for training in this run.",
            "runtime": {
                "requested": requested,
                "used": backend_used,
                "fallback_reason": fallback_reason,
            },
        }
    else:
        unsloth_section = {
            "used": False,
            "status": "not_used",
            "note": "Unsloth was not used in this run; PEFT path remained active.",
            "runtime": {
                "requested": requested,
                "used": backend_used,
                "fallback_reason": fallback_reason,
            },
        }

    payload = {
        "peft": {
            "used": backend_used == "peft",
            "retrieval_delta": retrieval_delta,
            "latency_delta_ms": latency_delta_ms,
            "rag_summary": rag_report,
            "graphrag_summary": graphrag_report,
        },
        "unsloth": unsloth_section,
        "trl": {
            "used_in_runtime": False,
            "status": "documentation_only",
            "reason": (
                "This project version targets encoder-style contrastive embedding alignment, "
                "so TRL trainers are not part of runtime execution."
            ),
        },
    }
    save_json(payload, settings.reports_dir / "tool_impact_report.json")
    return payload
