# Domain-Adapted Embedding Alignment

Production-oriented local pipeline for adapting embedding behavior across medical, legal, and cybersecurity retrieval workloads.

This repository prepares real multi-domain corpora, trains a compact embedding model with adapter-based fine-tuning, benchmarks sparse/dense/hybrid retrieval plus RAG/GraphRAG behavior, and emits auditable completion artifacts.

## What This Project Does

- Builds real `documents`, `queries`, and contrastive `pairs` artifacts from public datasets.
- Trains `Qwen/Qwen3-Embedding-0.6B` with LoRA using backend routing (`peft` or `unsloth`).
- Evaluates BM25, dense baseline, dense tuned, and hybrid retrieval with latency + quality metrics.
- Benchmarks Chroma and optional Pinecone RAG paths.
- Benchmarks GraphRAG local/global retrieval behavior.
- Generates machine-readable final reports and completion checks.

## Verified Run Snapshot (Ground Truth)

Source of truth:
- `artifacts/reports/data_preparation_report.json`
- `artifacts/reports/training_report.json`
- `artifacts/evaluation/retrieval_evaluation.json`
- `artifacts/evaluation/rag_benchmark.json`
- `artifacts/evaluation/graphrag_benchmark.json`
- `artifacts/reports/completion_checklist.json`

### Data Preparation
- Documents: **168,448**
- Queries: **98,907**
- Pairs: **200,000**
- Domain document mix: general 153,637 | medical 4,392 | legal 7,999 | cybersecurity 2,420

### Training
- Backend requested/used: **unsloth / unsloth**
- Total steps: **700**
- Best validation loss: **0.0694618**
- Train runtime: **5120.97s**
- Eval runtime: **231.18s**

### Retrieval (selected)
- `bm25`: recall@10 **0.9667**, MRR **0.7691**, mean latency **0.86ms**
- `baseline_dense_qwen4b`: recall@10 **1.0000**, MRR **0.8542**, mean latency **2902.25ms**
- `tuned_dense_qwen0_6b_lora`: recall@10 **0.9750**, MRR **0.7556**, mean latency **47.33ms**
- `hybrid_tuned`: recall@10 **0.9833**, MRR **0.8107**, mean latency **56.04ms**

Similarity diagnostics (`retrieval_evaluation.json`):
- Ranking accuracy: baseline **0.8933** -> tuned **0.9700**

### RAG / GraphRAG
- Chroma hit@5: baseline **1.0000**, tuned **0.9417**
- Pinecone: **disabled** in this run (`"executed": false`)
- GraphRAG local/global hit@5:
  - baseline **0.9467 / 0.4600**
  - tuned **0.8800 / 0.4600**

### Completion
- `project_completion_ok`: **true**

## Architecture Map

Core package: `src/domain_adapted_embedding_alignment/`

- `data/`: dataset loaders and pair mining
- `training/`: PEFT trainer, Unsloth trainer, backend resolver
- `retrieval/`: BM25, dense retriever, hybrid fusion, embedding backends
- `evaluation/`: metrics, judge integration, embedding-space diagnostics
- `rag/`: Chroma/Pinecone benchmark helpers and inference formatting
- `graphrag/`: graph construction and local/global retrieval
- `pipelines/`: stage orchestration and final report assembly
- CLI entrypoint: `cli.py`

## Setup (uv only)

```bash
cd /home/ahmad/AI/Domain-Adapted-Embedding-Alignment
uv python install 3.12.10
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

Optional extras:

```bash
uv sync --extra unsloth
uv sync --extra trl
```

## Runtime Requirements

Start Ollama:

```bash
ollama serve
```

Pull models used by the workflows:

```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
ollama pull glm-ocr
```

Optional Pinecone requires `PINECONE_API_KEY`.

## Run the Pipeline

Full pipeline:

```bash
dea-cli run-all
```

Stage-by-stage:

```bash
dea-cli prepare-data
dea-cli train --training-backend auto
dea-cli evaluate
dea-cli rag-benchmark
dea-cli graphrag-benchmark
dea-cli inference --query "What treatments are available for a brain tumor?"
dea-cli build-final-report
dea-cli finalize-reports
```

Notebook execution:

```bash
bash scripts/execute_notebooks.sh
```

## Key Artifacts

- Data and training:
  - `data/processed/documents.parquet`
  - `data/processed/queries.parquet`
  - `data/processed/pairs.parquet`
  - `artifacts/reports/training_report.json`
- Evaluation and benchmarks:
  - `artifacts/evaluation/retrieval_evaluation.json`
  - `artifacts/evaluation/benchmark_table.csv`
  - `artifacts/evaluation/rag_benchmark.json`
  - `artifacts/evaluation/graphrag_benchmark.json`
- Finalization:
  - `artifacts/reports/final_report.json`
  - `artifacts/reports/completion_checklist.json`
  - `artifacts/reports/tool_impact_report.json`

## Documentation

- Handbook: `docs/handbook.md`
- Tutorials: `docs/tutorials/00_learning_path.md` through `docs/tutorials/12_reports_validation_and_auditability.md`
- Tooling rationale: `docs/tooling_decisions.md`
- Unified bundle: `docs/documentation.md`
- PDF: `docs/documentation.pdf`

## Operational Notes

- Judge model parsing can fail intermittently; fallback judge scores are neutral by design (`evaluation/llm_judge.py`).
- Pinecone is optional and run-dependent.
- Latency numbers are environment/profile dependent (`settings.py`, `configs/local_6gb.yaml`).

