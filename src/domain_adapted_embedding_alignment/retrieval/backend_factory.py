"""Factory helpers for robust backend initialization."""

from __future__ import annotations

from loguru import logger

from domain_adapted_embedding_alignment.retrieval.embeddings import (
    HuggingFaceEmbeddingBackend,
    OllamaEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from domain_adapted_embedding_alignment.settings import Settings


def build_baseline_backend(settings: Settings):
    """Prefer Ollama qwen3-embedding:4b baseline, with explicit fallback."""
    if settings.use_ollama_embeddings:
        backend = OllamaEmbeddingBackend(model_name=settings.baseline_embedding_model)
        try:
            # Cheap preflight call ensures the model is available.
            _ = backend.embed_texts(["baseline preflight"], normalize=True)
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ollama baseline backend failed ({}). Falling back to HF base model {}.",
                exc,
                settings.trainable_model_name,
            )

    return HuggingFaceEmbeddingBackend(
        model_name=settings.trainable_model_name,
        adapter_path=None,
        max_length=settings.max_doc_length,
        batch_size=settings.eval_batch_size,
    )


def build_tuned_backend(settings: Settings):
    """Build tuned backend from best adapter, with ST fallback for Unsloth outputs."""
    best_path = settings.model_dir / "best_adapter"
    final_path = settings.model_dir / "final_adapter"

    # Preferred path: base model + PEFT adapter.
    try:
        return HuggingFaceEmbeddingBackend(
            model_name=settings.trainable_model_name,
            adapter_path=best_path,
            max_length=settings.max_doc_length,
            batch_size=settings.eval_batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PEFT tuned backend load failed: {}", exc)

    # Fallback path for SentenceTransformer-compatible exports.
    for candidate in [best_path, final_path]:
        if not candidate.exists():
            continue
        if (candidate / "modules.json").exists():
            try:
                return SentenceTransformerEmbeddingBackend(
                    model_name_or_path=str(candidate),
                    batch_size=settings.eval_batch_size,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SentenceTransformer tuned backend load failed for {}: {}", candidate, exc)

    # Last-resort explicit failure keeps debugging clear for users.
    raise RuntimeError(
        "Failed to load tuned backend. Checked PEFT adapter and SentenceTransformer-compatible outputs."
    )
