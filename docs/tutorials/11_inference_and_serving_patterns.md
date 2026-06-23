# 11 - Inference and Serving Patterns

## What it is
Query-time retrieval path that returns ranked document candidates with metadata and text previews.

## Why it is used
This is the production-facing interface of the system: users submit queries, retrieval returns explainable ranked context.

## How it appears in code
- Inference stage: `src/domain_adapted_embedding_alignment/pipelines/run_inference.py`
- Query execution helper: `src/domain_adapted_embedding_alignment/rag/inference.py`
- Tuned backend wiring: `build_tuned_backend(...)` in `src/domain_adapted_embedding_alignment/retrieval/backend_factory.py`

## Practical explanation from run outputs
From `artifacts/logs/05_inference.log`:
- Example query count: `3`
- Mean latency: `65.10ms`
- Backend used: `HuggingFaceEmbeddingBackend`
- Output structure includes:
  - `rank`
  - `doc_id`
  - `score`
  - `domain`
  - `source`
  - `preview`

## Beginner checkpoint
- Understand how top-k retrieval output differs from final answer generation.

## Advanced checkpoint
- Define inference logging schema for offline error analysis and relevance drift detection.
- Design caching policy for repeated queries without degrading freshness.

## Practical guidance
Persist query and top-k outputs in your serving layer so you can debug retrieval failures with evidence, not intuition.

