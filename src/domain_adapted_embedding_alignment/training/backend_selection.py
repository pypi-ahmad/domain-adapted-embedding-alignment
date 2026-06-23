"""Training backend selection for PEFT vs optional Unsloth acceleration."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

import torch


TrainingBackend = Literal["auto", "peft", "unsloth"]
ResolvedTrainingBackend = Literal["peft", "unsloth"]


@dataclass(slots=True)
class BackendResolution:
    """Resolved backend decision with a reason string for reporting."""

    requested: TrainingBackend
    used: ResolvedTrainingBackend
    fallback_reason: str | None
    unsloth_available: bool
    unsloth_reason: str


def unsloth_runtime_status() -> tuple[bool, str]:
    """Return whether Unsloth is runnable in the current environment."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if importlib.util.find_spec("unsloth") is None:
        return False, "unsloth package is not installed"
    if importlib.util.find_spec("sentence_transformers") is None:
        return False, "sentence-transformers package is not installed"
    return True, "available"


def resolve_training_backend(requested: TrainingBackend, strict_backend: bool) -> BackendResolution:
    """Resolve the effective training backend with safe fallback semantics."""
    if requested not in {"auto", "peft", "unsloth"}:
        raise ValueError(f"Unsupported training backend: {requested}")

    unsloth_available, unsloth_reason = unsloth_runtime_status()

    if requested == "peft":
        return BackendResolution(
            requested=requested,
            used="peft",
            fallback_reason=None,
            unsloth_available=unsloth_available,
            unsloth_reason=unsloth_reason,
        )

    if requested == "unsloth":
        if unsloth_available:
            return BackendResolution(
                requested=requested,
                used="unsloth",
                fallback_reason=None,
                unsloth_available=True,
                unsloth_reason=unsloth_reason,
            )
        if strict_backend:
            raise RuntimeError(
                f"Requested backend='unsloth' but runtime requirements are not met: {unsloth_reason}"
            )
        return BackendResolution(
            requested=requested,
            used="peft",
            fallback_reason=unsloth_reason,
            unsloth_available=False,
            unsloth_reason=unsloth_reason,
        )

    # requested == "auto"
    if unsloth_available:
        return BackendResolution(
            requested=requested,
            used="unsloth",
            fallback_reason=None,
            unsloth_available=True,
            unsloth_reason=unsloth_reason,
        )

    return BackendResolution(
        requested=requested,
        used="peft",
        fallback_reason=unsloth_reason,
        unsloth_available=False,
        unsloth_reason=unsloth_reason,
    )
