# 09 - RAG with Chroma and Pinecone

## What it is
Benchmark stage for retrieval-augmented generation contexts using:
- local Chroma collections
- optional Pinecone indexes

## Why it is used
RAG systems need stable top-k retrieval quality and predictable latency under realistic index/query workloads.

## How it appears in code
- Pipeline: `src/domain_adapted_embedding_alignment/pipelines/run_rag_benchmarks.py`
- Chroma indexing/search: `src/domain_adapted_embedding_alignment/rag/chroma_demo.py`
- Pinecone indexing/search: `src/domain_adapted_embedding_alignment/rag/pinecone_demo.py`

## Practical explanation from run outputs
From `artifacts/logs/03_rag.log`:
- Chroma baseline index built for 1500 docs
- Chroma tuned index built for 1500 docs

From `artifacts/evaluation/rag_benchmark.json`:
- Chroma baseline hit@5: `1.0`
- Chroma tuned hit@5: `0.9416666666666667`
- Chroma mean latency: baseline `159.95ms`, tuned `45.06ms`
- Pinecone: `executed=false` (`reason: disabled`)

## Beginner checkpoint
- Understand difference between local vector store and managed vector DB.

## Advanced checkpoint
- Decide when Pinecone should be mandatory instead of optional.
- Define SLOs using p95/p99 latency and hit-rate thresholds.

## Operational note
Pinecone execution requires runtime enablement plus credentials.

