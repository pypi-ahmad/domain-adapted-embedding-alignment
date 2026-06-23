"""Notebook structure checks for tutorial completeness."""

from pathlib import Path

import nbformat


REQUIRED_HEADINGS = [
    "## What is This Technique?",
    "## Definition and Core Concepts",
    "## Why Was This Technique Developed?",
    "## What Limitations of Traditional RAG Does It Solve?",
    "## Architecture and Workflow Diagram Explanation",
    "## Component-by-Component Breakdown",
    "## When Should It Be Used in Real-World Systems?",
    "## Advantages and Disadvantages",
    "## Comparison Against Standard RAG and Implemented RAG Variants",
    "## Implementation Details and Design Decisions Used in This Project",
    "## Post-Run Analysis of Actual Outputs and Metrics",
]


def test_all_notebooks_have_required_sections() -> None:
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
    paths = sorted(notebook_dir.glob("*.ipynb"))
    assert paths, "No notebooks found."

    for path in paths:
        nb = nbformat.read(path, as_version=4)
        markdown_text = "\n".join(cell["source"] for cell in nb.cells if cell["cell_type"] == "markdown")
        for heading in REQUIRED_HEADINGS:
            assert heading in markdown_text, f"{path.name} is missing heading: {heading}"
