"""Command-line interface for the Domain-Adapted Embedding Alignment project."""

from __future__ import annotations

import argparse
import json

from domain_adapted_embedding_alignment.pipelines.build_final_report import run_build_final_report
from domain_adapted_embedding_alignment.pipelines.evaluate_models import run_evaluation
from domain_adapted_embedding_alignment.pipelines.finalize_reports import (
    build_completion_checklist,
    build_tool_impact_report,
)
from domain_adapted_embedding_alignment.pipelines.prepare_data import run_prepare_data
from domain_adapted_embedding_alignment.pipelines.run_end_to_end import run_end_to_end
from domain_adapted_embedding_alignment.pipelines.run_graphrag_benchmarks import run_graphrag_benchmarks
from domain_adapted_embedding_alignment.pipelines.run_inference import run_demo_inference
from domain_adapted_embedding_alignment.pipelines.run_rag_benchmarks import run_rag_benchmarks
from domain_adapted_embedding_alignment.pipelines.train_model import run_train
from domain_adapted_embedding_alignment.settings import get_settings
from domain_adapted_embedding_alignment.utils import configure_logging, set_seed


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    configure_logging()
    settings = get_settings()
    set_seed(settings.random_seed)

    parser = argparse.ArgumentParser(description="Domain-Adapted Embedding Alignment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Load datasets and build pair/triplet artifacts")
    train_parser = sub.add_parser("train", help="Fine-tune embedding model with contrastive learning")
    train_parser.add_argument(
        "--training-backend",
        choices=["auto", "peft", "unsloth"],
        default=None,
        help="Override training backend. Defaults to DEA_TRAINING_BACKEND / settings value.",
    )

    eval_parser = sub.add_parser("evaluate", help="Run baseline vs fine-tuned retrieval evaluation")
    eval_parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-judge scoring")

    sub.add_parser("rag-benchmark", help="Run ChromaDB and Pinecone benchmark demos")
    sub.add_parser("graphrag-benchmark", help="Run GraphRAG local/global retrieval benchmarks")
    sub.add_parser("build-final-report", help="Assemble final report from existing artifacts + inference")
    sub.add_parser("finalize-reports", help="Build completion checklist and tool impact reports")

    inf_parser = sub.add_parser("inference", help="Run inference examples")
    inf_parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query to run (pass multiple times for multiple queries)",
    )

    end_parser = sub.add_parser("run-all", help="Run full pipeline end-to-end")
    end_parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-judge during evaluation")
    end_parser.add_argument(
        "--training-backend",
        choices=["auto", "peft", "unsloth"],
        default=None,
        help="Override training backend for the training stage.",
    )

    args = parser.parse_args()

    if args.command == "prepare-data":
        _print_json(run_prepare_data(settings))
    elif args.command == "train":
        _print_json(run_train(settings, training_backend=args.training_backend))
    elif args.command == "evaluate":
        _print_json(run_evaluation(settings, run_judge=not args.no_judge))
    elif args.command == "rag-benchmark":
        _print_json(run_rag_benchmarks(settings))
    elif args.command == "graphrag-benchmark":
        _print_json(run_graphrag_benchmarks(settings))
    elif args.command == "build-final-report":
        _print_json(run_build_final_report(settings))
    elif args.command == "finalize-reports":
        _print_json(
            {
                "completion": build_completion_checklist(settings),
                "tool_impact": build_tool_impact_report(settings),
            }
        )
    elif args.command == "inference":
        queries = args.query or [
            "What treatments are available for a brain tumor?",
            "What are legal grounds for termination of contract?",
            "How does credential theft relate to account compromise techniques?",
        ]
        _print_json(run_demo_inference(settings, queries=queries))
    elif args.command == "run-all":
        _print_json(
            run_end_to_end(
                settings,
                run_judge=not args.no_judge,
                training_backend=args.training_backend,
            )
        )


if __name__ == "__main__":
    main()
