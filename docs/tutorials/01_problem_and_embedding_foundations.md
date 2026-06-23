# 01 - Problem and Embedding Foundations

## What it is
Domain-adapted embedding alignment means training an embedding model so semantically related items in your specific domain are closer in vector space than irrelevant but lexically similar items.

## Why it is used
General-purpose embeddings can miss domain nuance. Terms that look close in generic corpora are not always close for downstream retrieval objectives in medicine, law, or cybersecurity.

## How it appears in code
- Model and runtime defaults: `src/domain_adapted_embedding_alignment/settings.py`
- Embedding backend implementations: `src/domain_adapted_embedding_alignment/retrieval/embeddings.py`
- Baseline/tuned backend routing: `src/domain_adapted_embedding_alignment/retrieval/backend_factory.py`
- Training objective entrypoints:
  - `src/domain_adapted_embedding_alignment/training/trainer.py`
  - `src/domain_adapted_embedding_alignment/training/unsloth_trainer.py`

## Practical explanation from run outputs
From `artifacts/evaluation/retrieval_evaluation.json`:
- Baseline dense quality is stronger on top-level recall/MRR in this run.
- Tuned dense is dramatically faster (`47.33ms` mean vs `2902.25ms` mean).
- Similarity ranking accuracy improved (`0.8933 -> 0.9700`) for sampled pair diagnostics.

This is the core lesson: adaptation is a systems tradeoff problem, not a single-metric contest.

## Beginner checkpoint
- Understand cosine similarity at a conceptual level.
- Understand why positive and hard-negative pairs are needed.
- Understand why you must report both quality and latency.

## Advanced checkpoint
- Explain why retrieval metrics and pairwise ranking diagnostics can disagree.
- Explain why backend choice (`ollama` baseline vs tuned HF adapter) can dominate latency behavior.

