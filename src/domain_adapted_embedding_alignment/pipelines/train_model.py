"""Fine-tuning pipeline for domain-aligned embedding model."""

from __future__ import annotations

from loguru import logger

from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.training.backend_selection import resolve_training_backend
from domain_adapted_embedding_alignment.training.trainer import ContrastiveTrainer
from domain_adapted_embedding_alignment.training.unsloth_trainer import UnslothContrastiveTrainer
from domain_adapted_embedding_alignment.utils import save_json


def run_train(settings: Settings, training_backend: str | None = None) -> dict:
    """Run contrastive embedding alignment with backend-aware training."""
    pairs_path = settings.processed_data_dir / "pairs.parquet"
    if not pairs_path.exists():
        raise FileNotFoundError("pairs.parquet missing. Run data preparation first.")

    requested_backend = settings.training_backend if training_backend is None else training_backend
    resolution = resolve_training_backend(
        requested=requested_backend,
        strict_backend=settings.strict_backend,
    )

    if resolution.used == "unsloth":
        logger.info("Starting model training with backend='unsloth'")
        trainer = UnslothContrastiveTrainer(settings=settings, pairs_path=pairs_path)
        history = trainer.train()
    else:
        logger.info("Starting model training with backend='peft'")
        trainer = ContrastiveTrainer(settings=settings, pairs_path=pairs_path)
        history = trainer.train()

    history["runtime"] = {
        "backend_requested": resolution.requested,
        "backend_used": resolution.used,
        "fallback_reason": resolution.fallback_reason,
        "unsloth_available": resolution.unsloth_available,
        "unsloth_reason": resolution.unsloth_reason,
    }

    save_json(history, settings.reports_dir / "training_report.json")
    logger.info("Training completed")
    return history
