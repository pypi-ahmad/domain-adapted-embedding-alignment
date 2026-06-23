"""Tests for training backend selection and fallback behavior."""

import pytest

from domain_adapted_embedding_alignment.training import backend_selection as bs


def test_auto_falls_back_to_peft_when_unsloth_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(bs, "unsloth_runtime_status", lambda: (False, "CUDA is not available"))
    resolution = bs.resolve_training_backend(requested="auto", strict_backend=False)
    assert resolution.used == "peft"
    assert resolution.fallback_reason == "CUDA is not available"


def test_unsloth_selected_when_available(monkeypatch) -> None:
    monkeypatch.setattr(bs, "unsloth_runtime_status", lambda: (True, "available"))
    resolution = bs.resolve_training_backend(requested="unsloth", strict_backend=False)
    assert resolution.used == "unsloth"
    assert resolution.fallback_reason is None


def test_unsloth_strict_mode_raises_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(bs, "unsloth_runtime_status", lambda: (False, "unsloth package is not installed"))
    with pytest.raises(RuntimeError):
        bs.resolve_training_backend(requested="unsloth", strict_backend=True)


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError):
        bs.resolve_training_backend(requested="invalid", strict_backend=False)  # type: ignore[arg-type]
