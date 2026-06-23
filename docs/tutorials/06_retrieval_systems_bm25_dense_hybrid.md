# 06 - Retrieval Systems: BM25, Dense, Hybrid

## What it is
A comparative retrieval layer with sparse lexical search, dense semantic search, and hybrid fusion.

## Why it is used
No single retrieval method consistently dominates across all domains, latency budgets, and query distributions.

## How it appears in code
- Sparse retriever: `src/domain_adapted_embedding_alignment/retrieval/bm25.py`
- Dense retriever: `src/domain_adapted_embedding_alignment/retrieval/dense.py`
- Hybrid fusion functions: `src/domain_adapted_embedding_alignment/retrieval/hybrid.py`
  - `weighted_score_fusion`
  - `reciprocal_rank_fusion`
- Evaluation orchestration: `src/domain_adapted_embedding_alignment/pipelines/evaluate_models.py`

## Practical explanation from run outputs
From `artifacts/evaluation/retrieval_evaluation.json`:
- BM25: recall@10 `0.9667`, mean latency `0.86ms`
- Baseline dense: recall@10 `1.0000`, MRR `0.8542`, mean latency `2902.25ms`
- Tuned dense: recall@10 `0.9750`, MRR `0.7556`, mean latency `47.33ms`
- Hybrid tuned: recall@10 `0.9833`, MRR `0.8107`, mean latency `56.04ms`

Interpretation: baseline dense provides highest quality in this profile; tuned dense and hybrid tuned offer much lower latency.

## Beginner checkpoint
- Understand lexical vs semantic retrieval behavior.
- Understand why latency must be reported with quality.

## Advanced checkpoint
- Evaluate whether hybrid should be weighted differently by domain.
- Identify tail-latency outliers (`max_ms`) and potential causes.

## Practical takeaway
Use `artifacts/evaluation/benchmark_table.csv` as deployment guidance, not leaderboard-only scoring.

