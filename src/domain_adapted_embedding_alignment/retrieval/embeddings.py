"""Embedding backends for baseline and fine-tuned retrieval.

Backends included:
- Ollama backend (for qwen3-embedding:4b baseline and judge-time embeddings)
- Hugging Face backend (for fine-tuned Qwen3-Embedding-0.6B + LoRA adapters)
- SentenceTransformers backend (for Unsloth/SentenceTransformer-compatible outputs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import ollama
import torch
import torch.nn.functional as F
from loguru import logger
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from domain_adapted_embedding_alignment.utils import batched


class EmbeddingBackend(Protocol):
    """Minimal embedding backend contract used across retrieval pipelines."""

    def embed_texts(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Embed a batch of texts and return a 2D NumPy array."""


class OllamaEmbeddingBackend:
    """Embedding backend backed by a local Ollama model."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        vectors: list[list[float]] = []
        for chunk in batched(texts, self.batch_size):
            try:
                response = ollama.embed(model=self.model_name, input=chunk)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to embed texts with Ollama model '{self.model_name}'. "
                    "Ensure `ollama serve` is running and model is pulled."
                ) from exc

            embeddings = response.get("embeddings", [])
            if not embeddings:
                raise RuntimeError(f"Ollama returned empty embeddings for model '{self.model_name}'")
            vectors.extend(embeddings)

        matrix = np.asarray(vectors, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            matrix = matrix / norms
        return matrix


class HuggingFaceEmbeddingBackend:
    """HF embedding backend for base or LoRA-adapted Qwen embedding models."""

    def __init__(
        self,
        model_name: str,
        adapter_path: Path | None = None,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info("Loading HF embedding backend model='{}' on {}", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float32 if self.device == "cpu" else None
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        if adapter_path is not None and adapter_path.exists():
            logger.info("Loading LoRA adapter from {}", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_texts(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        all_vectors: list[np.ndarray] = []

        for chunk in batched(texts, self.batch_size):
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)

            # Mean-pooling over valid tokens.
            summed = (last_hidden_state * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            embeddings = summed / counts

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_vectors.append(embeddings.float().detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_vectors)


class SentenceTransformerEmbeddingBackend:
    """SentenceTransformers embedding backend for merged/finalized ST-compatible models."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info("Loading SentenceTransformer backend model='{}' on {}", model_name_or_path, self.device)
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

    def embed_texts(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
