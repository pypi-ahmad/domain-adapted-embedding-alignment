"""Utility helpers for logging, reproducibility, and artifact I/O."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from loguru import logger


def configure_logging() -> None:
    """Configure loguru to emit structured logs to stderr by default."""
    logger.remove()
    use_json = os.getenv("DEA_LOG_JSON", "true").strip().lower() in {"1", "true", "yes", "on"}
    if use_json:
        logger.add(sys.stderr, serialize=True, backtrace=False, diagnose=False)
        return

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        backtrace=False,
        diagnose=False,
    )


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: Any, path: Path) -> None:
    """Persist JSON payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Load JSON payload from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    """Yield fixed-size batches from a list."""
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def timed(label: str):
    """Simple timing decorator for pipeline stages."""

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{label} finished in {elapsed:.2f}s")
            return result

        return _wrapper

    return _decorator


def latency_summary(latencies_seconds: list[float]) -> dict[str, float]:
    """Summarize latencies as milliseconds for consistent report payloads."""
    if not latencies_seconds:
        return {
            "count": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }

    arr = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    return {
        "count": float(arr.shape[0]),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
    }
