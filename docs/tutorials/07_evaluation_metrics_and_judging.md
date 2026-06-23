# 07 - Evaluation Metrics and LLM Judging

## What it is
Evaluation layer that combines:
- retrieval quality metrics (`recall@k`, `precision@k`, `MRR`, `NDCG`, `MAP`)
- latency summaries (`mean`, `p50`, `p95`, `max`)
- RAG proxy metrics (`context_precision`, `context_recall`)
- optional LLM judge metrics
- embedding-space diagnostics (ranking accuracy and clustering summaries)

## Why it is used
A retrieval system can look excellent on one metric and fail operationally or semantically. Multi-view evaluation prevents false confidence.

## How it appears in code
- Evaluator core: `src/domain_adapted_embedding_alignment/evaluation/evaluator.py`
- Metric implementation: `src/domain_adapted_embedding_alignment/retrieval/metrics.py`
- Judge calls and fallback: `src/domain_adapted_embedding_alignment/evaluation/llm_judge.py`
- Stage wrapper: `src/domain_adapted_embedding_alignment/pipelines/evaluate_models.py`

## Practical explanation from run outputs
From `artifacts/evaluation/retrieval_evaluation.json`:
- Judge metrics are present for baseline and tuned dense systems.
- Similarity ranking accuracy improved from `0.8933` to `0.9700`.

From `artifacts/logs/02_evaluate.log`:
- Repeated parse failures occurred for judge model `qwen3.5:4b`.
- Fallback behavior returned neutral scores so evaluation completed successfully.

## Beginner checkpoint
- Understand what each retrieval metric measures.
- Understand why judge scores are auxiliary diagnostics.

## Advanced checkpoint
- Identify when judge fallback should invalidate model comparisons.
- Define confidence intervals or bootstrap plans for retrieval metrics across domains.

## Important interpretation note
Judge output is helpful but non-authoritative. Primary retrieval decisions should still rely on deterministic retrieval and latency metrics.

