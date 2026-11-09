"""UMAP projections and clustering over the stored vectors (nothing is re-embedded).

Replaces the original playbook's PCA plot and its threshold-based "clusters" (any two
videos closer than the 25th-percentile distance) with UMAP (cosine) and K-Means or
HDBSCAN. Clustering runs on the 512-d vectors, not on the 2-D picture, so the
picture only shows the clusters; it doesn't make them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SEED = 42
KMEANS, HDBSCAN = "K-Means", "HDBSCAN"
ALGORITHMS = (KMEANS, HDBSCAN)
NOISE = -1

# lancedb.com's orange and lavender lead; the rest are spaced for contrast on dark and light.
PALETTE = ("#FF734A", "#9085E9", "#199E70", "#C98500", "#3987E5", "#D55181", "#AD9E95", "#008300")
NOISE_COLOR = "#8A817C"


def cluster_color(label: int) -> str:
    return NOISE_COLOR if label == NOISE else PALETTE[label % len(PALETTE)]


def stack(values: Iterable[object]) -> np.ndarray:
    rows = [np.asarray(v, dtype=np.float32) for v in values]
    return np.stack(rows) if rows else np.empty((0, 512), np.float32)


def project(vectors: np.ndarray, n_components: int = 2, n_neighbors: int = 10) -> np.ndarray:
    """UMAP with the cosine metric; PCA for inputs too small for UMAP."""
    n = len(vectors)
    if n <= n_components + 2:
        centered = vectors - vectors.mean(axis=0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        out = np.zeros((n, n_components), np.float32)
        k = min(n_components, u.shape[1])
        out[:, :k] = (u * s)[:, :k]
        return out
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, n - 1),
        metric="cosine",
        random_state=SEED,
        n_jobs=1,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def cluster(vectors: np.ndarray, algorithm: str, k: int, min_size: int) -> np.ndarray:
    """Cluster labels for the (normalised) 512-d vectors; HDBSCAN marks noise as -1."""
    n = len(vectors)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    if algorithm == KMEANS:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=min(k, n), random_state=SEED, n_init=10).fit_predict(vectors)
    from sklearn.cluster import HDBSCAN as Hdbscan

    return Hdbscan(min_cluster_size=max(2, min(min_size, n)), metric="cosine", copy=True).fit_predict(vectors)


def centroid(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def cohesion(vectors: np.ndarray) -> float:
    """Mean cosine similarity of a cluster's members to their centroid."""
    return float((vectors @ centroid(vectors)).mean()) if len(vectors) else 0.0


def outliers(vectors: np.ndarray, ids: list[str], top: int = 5) -> pd.DataFrame:
    """The members least like everything else: lowest mean cosine similarity to the others."""
    n = len(vectors)
    if n < 3:
        return pd.DataFrame(columns=["id", "mean similarity to others"])
    sims = vectors @ vectors.T
    mean_other = (sims.sum(axis=1) - np.diag(sims)) / (n - 1)
    order = np.argsort(mean_other)[:top]
    return pd.DataFrame({"id": [ids[i] for i in order], "mean similarity to others": mean_other[order].round(4)})
