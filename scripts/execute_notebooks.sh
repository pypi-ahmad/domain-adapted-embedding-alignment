#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

for nb in \
  notebooks/01_problem_analysis_and_embedding_theory.ipynb \
  notebooks/02_data_preparation_and_pair_mining.ipynb \
  notebooks/03_baseline_retrieval_systems.ipynb \
  notebooks/04_contrastive_learning_and_finetuning.ipynb \
  notebooks/05_evaluation_and_error_analysis.ipynb \
  notebooks/06_visualization_embedding_space.ipynb \
  notebooks/07_rag_integration_chroma_pinecone.ipynb \
  notebooks/08_graphrag_integration_and_inference.ipynb
  do
  echo "Executing $nb"
  out_name="$(basename "${nb%.ipynb}").executed.ipynb"
  jupyter nbconvert --to notebook --execute "$nb" --output "$out_name" --output-dir notebooks
done
