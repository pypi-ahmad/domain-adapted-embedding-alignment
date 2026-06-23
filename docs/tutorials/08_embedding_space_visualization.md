# 08 - Embedding Space Visualization

## What it is
Projection of high-dimensional embeddings into 2D using PCA, t-SNE, and UMAP for qualitative diagnostics.

## Why it is used
Ranking metrics capture outcome quality but not geometric behavior. Projection plots can reveal clustering, overlap, or collapse patterns by domain.

## How it appears in code
- Visualization utilities: `src/domain_adapted_embedding_alignment/evaluation/visualization.py`
- Stage integration: `src/domain_adapted_embedding_alignment/pipelines/run_end_to_end.py`
  - `plot_embedding_projections(..., prefix="baseline")`
  - `plot_embedding_projections(..., prefix="tuned")`

## Practical explanation from run outputs
Generated outputs:
- `artifacts/figures/baseline_pca.png`
- `artifacts/figures/baseline_tsne.png`
- `artifacts/figures/baseline_umap.png`
- `artifacts/figures/tuned_pca.png`
- `artifacts/figures/tuned_tsne.png`
- `artifacts/figures/tuned_umap.png`

These figures are also validated by completion checks (`completion_checklist.json`).

## Beginner checkpoint
- Understand why PCA is more stable and t-SNE/UMAP more local-structure focused.

## Advanced checkpoint
- Compare projection narratives against quantitative metrics to avoid visual over-interpretation.

