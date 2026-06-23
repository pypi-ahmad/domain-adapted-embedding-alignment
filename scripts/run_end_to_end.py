#!/usr/bin/env python3
"""Convenience wrapper for full project execution."""

from domain_adapted_embedding_alignment.pipelines.run_end_to_end import run_end_to_end
from domain_adapted_embedding_alignment.settings import get_settings
from domain_adapted_embedding_alignment.utils import configure_logging, set_seed


if __name__ == "__main__":
    configure_logging()
    settings = get_settings()
    set_seed(settings.random_seed)
    run_end_to_end(settings)
