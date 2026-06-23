# 10 - GraphRAG Local and Global Retrieval

## What it is
A graph-augmented retrieval layer that adds two retrieval modes on top of dense seeds:
- local neighborhood expansion
- global community retrieval

## Why it is used
Dense retrieval alone can miss structural relationships. Graph-based expansion can inject useful context and improve retrieval behavior in some settings.

## How it appears in code
- Graph construction: `src/domain_adapted_embedding_alignment/graphrag/graph_builder.py`
- Local graph retrieval: `local_graph_retrieval(...)` in `src/domain_adapted_embedding_alignment/graphrag/retriever.py`
- Global community retrieval: `global_community_retrieval(...)` in `src/domain_adapted_embedding_alignment/graphrag/retriever.py`
- Benchmark pipeline: `src/domain_adapted_embedding_alignment/pipelines/run_graphrag_benchmarks.py`

## Practical explanation from run outputs
From `artifacts/logs/04_graphrag.log`:
- Graph built with `9220` nodes, `187729` edges, `6` communities

From `artifacts/evaluation/graphrag_benchmark.json`:
- Baseline local/global hit@5: `0.9467 / 0.46`
- Tuned local/global hit@5: `0.88 / 0.46`
- Tuned dense latency mean: `54.48ms` (vs baseline `157.65ms`)

## Beginner checkpoint
- Understand the difference between local expansion and global community retrieval.

## Advanced checkpoint
- Assess whether global retrieval should include embedding features beyond term overlap.
- Evaluate graph construction sensitivity to term extraction heuristics.

## Practical takeaway
In this run, local and dense latency improved for tuned path, but global hit@5 remained unchanged.

