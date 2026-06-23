# 00 - Learning Path

## What it is
A structured route through the repository from first principles to production-style operation.

## Why it is used
This project spans data engineering, contrastive training, retrieval science, benchmarking, and audit reporting. A defined sequence prevents fragmented understanding.

## How it appears in code
- Workflow entrypoint: `src/domain_adapted_embedding_alignment/cli.py`
- Stage orchestration: `src/domain_adapted_embedding_alignment/pipelines/run_end_to_end.py`
- Executable notebook track:
  - `notebooks/01_problem_analysis_and_embedding_theory.ipynb`
  - `notebooks/02_data_preparation_and_pair_mining.ipynb`
  - `notebooks/03_baseline_retrieval_systems.ipynb`
  - `notebooks/04_contrastive_learning_and_finetuning.ipynb`
  - `notebooks/05_evaluation_and_error_analysis.ipynb`
  - `notebooks/06_visualization_embedding_space.ipynb`
  - `notebooks/07_rag_integration_chroma_pinecone.ipynb`
  - `notebooks/08_graphrag_integration_and_inference.ipynb`

## Practical path
1. Read `README.md` for the system picture and run snapshot.
2. Read `docs/handbook.md` for implementation depth.
3. Complete tutorials `01` to `05` to master data + training internals.
4. Complete tutorials `06` to `10` to master retrieval/RAG/GraphRAG behavior.
5. Complete tutorials `11` and `12` to understand inference and completion gates.

## Real run anchor points
Use these artifacts as your ground truth while studying:
- Data: `168448` docs, `98907` queries, `200000` pairs
- Training: `700` steps, backend used `unsloth`
- Completion: `project_completion_ok=true`

Files:
- `artifacts/reports/data_preparation_report.json`
- `artifacts/reports/training_report.json`
- `artifacts/reports/completion_checklist.json`

## Next tutorial
Proceed to `docs/tutorials/01_problem_and_embedding_foundations.md`.

