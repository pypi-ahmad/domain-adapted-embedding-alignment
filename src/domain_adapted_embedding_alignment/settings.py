"""Centralized configuration for reproducible local runs.

The project uses profile-driven settings so the same code can support:
- local smoke tests,
- the 6GB production profile,
- and larger follow-up training runs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with environment-variable overrides.

    Environment variables use the prefix ``DEA_``.
    Example: ``DEA_TRAIN_MAX_STEPS=800``.
    """

    model_config = SettingsConfigDict(env_prefix="DEA_", env_file=".env", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    # Reproducibility.
    random_seed: int = 17

    # Execution profile.
    profile_name: str = "local_6gb"
    use_ollama_embeddings: bool = True
    use_live_pinecone: bool = True
    skip_data_preparation_if_present: bool = False
    training_backend: Literal["auto", "peft", "unsloth"] = "auto"
    strict_backend: bool = False

    # Data budget.
    msmarco_max_rows: int = 160_000
    medical_max_records: int = 14_000
    legal_max_records: int = 8_000
    cyber_max_records: int = 18_000
    final_pair_target: int = 200_000

    # Model choices.
    baseline_embedding_model: str = "qwen3-embedding:4b"
    trainable_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    judge_model_primary: str = "granite4.1:3b"
    judge_model_secondary: str = "qwen3.5:4b"

    # Training controls (6GB-safe defaults).
    max_query_length: int = 96
    max_doc_length: int = 192
    train_batch_size: int = 2
    eval_batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    train_epochs: int = 1
    train_max_steps: int = 700
    eval_every_steps: int = 100
    save_every_steps: int = 100
    temperature: float = 0.05
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    unsloth_max_seq_length: int = 192
    unsloth_load_in_4bit: bool = True
    unsloth_val_max_rows: int = 2000

    # Retrieval / benchmarking.
    retrieval_top_k: int = 10
    rag_context_top_k: int = 6
    graph_neighbor_hops: int = 1
    judge_max_queries: int = 80
    eval_doc_limit: int = 1200
    eval_query_limit: int = 120
    similarity_sample_limit: int = 300
    rag_doc_limit: int = 1500
    rag_query_limit: int = 120
    graphrag_doc_limit: int = 1800
    graphrag_query_limit: int = 150
    inference_doc_limit: int = 2000

    # Paths.
    raw_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    artifacts_dir: Path | None = None
    model_dir: Path | None = None
    eval_dir: Path | None = None
    figures_dir: Path | None = None
    reports_dir: Path | None = None
    chroma_dir: Path | None = None

    @model_validator(mode="after")
    def _set_default_paths(self) -> "Settings":
        if self.raw_data_dir is None:
            self.raw_data_dir = self.project_root / "data" / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.project_root / "data" / "processed"
        if self.artifacts_dir is None:
            self.artifacts_dir = self.project_root / "artifacts"
        if self.model_dir is None:
            self.model_dir = self.artifacts_dir / "models"
        if self.eval_dir is None:
            self.eval_dir = self.artifacts_dir / "evaluation"
        if self.figures_dir is None:
            self.figures_dir = self.artifacts_dir / "figures"
        if self.reports_dir is None:
            self.reports_dir = self.artifacts_dir / "reports"
        if self.chroma_dir is None:
            self.chroma_dir = self.project_root / "chroma_db"
        return self

    def ensure_directories(self) -> None:
        """Create all required project directories."""
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
            self.model_dir,
            self.eval_dir,
            self.figures_dir,
            self.reports_dir,
            self.chroma_dir,
        ]:
            if path is None:
                continue
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object for consistent runtime config."""
    settings = Settings()
    settings.ensure_directories()
    return settings
