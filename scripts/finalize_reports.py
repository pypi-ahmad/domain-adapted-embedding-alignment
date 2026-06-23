#!/usr/bin/env python3
"""Generate completion and tooling impact reports."""

from domain_adapted_embedding_alignment.pipelines.finalize_reports import (
    build_completion_checklist,
    build_tool_impact_report,
)
from domain_adapted_embedding_alignment.settings import get_settings
from domain_adapted_embedding_alignment.utils import configure_logging, set_seed


if __name__ == "__main__":
    configure_logging()
    settings = get_settings()
    set_seed(settings.random_seed)
    completion = build_completion_checklist(settings)
    impact = build_tool_impact_report(settings)
    print({"completion": completion, "tool_impact": impact})
