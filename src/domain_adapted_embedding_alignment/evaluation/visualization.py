"""Visualization helpers for embedding space diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


try:
    import umap
except Exception:  # noqa: BLE001
    umap = None


def _scatter_plot(points: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", n_colors=max(3, len(unique_labels)))
    label_to_color = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 7))
    for label in unique_labels:
        mask = [idx for idx, value in enumerate(labels) if value == label]
        subset = points[mask]
        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            s=20,
            alpha=0.8,
            label=label,
            color=label_to_color[label],
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_embedding_projections(
    embeddings: np.ndarray,
    labels: list[str],
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    """Generate PCA, t-SNE, and UMAP figures for embedding comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(embeddings.shape[0])

    pca = PCA(n_components=2, random_state=17)
    pca_points = pca.fit_transform(embeddings)
    pca_path = output_dir / f"{prefix}_pca.png"
    _scatter_plot(pca_points, labels, f"{prefix.upper()} - PCA", pca_path)

    tsne_perplexity = float(max(2, min(30, n_samples - 1)))
    tsne = TSNE(
        n_components=2,
        random_state=17,
        init="pca",
        learning_rate="auto",
        perplexity=tsne_perplexity,
    )
    tsne_points = tsne.fit_transform(embeddings)
    tsne_path = output_dir / f"{prefix}_tsne.png"
    _scatter_plot(tsne_points, labels, f"{prefix.upper()} - t-SNE", tsne_path)

    umap_path = output_dir / f"{prefix}_umap.png"
    if umap is not None:
        umap_neighbors = int(max(2, min(20, n_samples - 1)))
        reducer = umap.UMAP(
            n_neighbors=umap_neighbors,
            min_dist=0.2,
            metric="cosine",
            random_state=17,
        )
        umap_points = reducer.fit_transform(embeddings)
        _scatter_plot(umap_points, labels, f"{prefix.upper()} - UMAP", umap_path)
    else:
        # If UMAP import fails, fallback to PCA visualization while preserving output contract.
        _scatter_plot(pca_points, labels, f"{prefix.upper()} - UMAP fallback", umap_path)

    return {
        "pca": str(pca_path),
        "tsne": str(tsne_path),
        "umap": str(umap_path),
    }
