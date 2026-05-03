"""KMeans backend switcher — sklearn (default) or cuml (GPU).

Set ``SELFCHAT_KMEANS=cuml`` to route through ``cuml.cluster.KMeans``;
falls back to sklearn if cuml isn't installed. Sparse inputs are
densified to ``float32`` for cuml (fine for ~7k × ~50k TF-IDF on a 5090;
warns past 4 GiB).

Usage:
    from selfchat.stats._kmeans import fit_kmeans
    labels = fit_kmeans(X, k, seed)
"""

from __future__ import annotations

import os

import numpy as np


_BACKEND: tuple[str, type] | None = None
K_SWEEP = list(range(2, 21)) + [25, 30, 35, 40, 45, 50]
SIZE_FLOOR_DEFAULT = 30


def _resolve_backend() -> tuple[str, type]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    want = os.environ.get("SELFCHAT_KMEANS", "").lower()
    if want == "cuml":
        try:
            from cuml.cluster import KMeans as CuMLKMeans  # type: ignore[import-not-found, import-untyped]
            _BACKEND = ("cuml", CuMLKMeans)
            print("[selfchat] kmeans backend: cuml (GPU)")
            return _BACKEND
        except ImportError:
            print(
                "[selfchat] SELFCHAT_KMEANS=cuml requested but cuml not "
                "installed; falling back to sklearn"
            )
    from sklearn.cluster import KMeans as SkKMeans  # type: ignore[import-not-found]
    _BACKEND = ("sklearn", SkKMeans)
    return _BACKEND


_SPARSE_FALLBACK_NOTICED = False


def fit_kmeans(X, k: int, seed: int, n_init: int = 10) -> np.ndarray:
    """Fit KMeans and return labels. Backend per ``SELFCHAT_KMEANS``.

    cuml requires dense input, but densifying high-dim TF-IDF (~150k cols)
    is dramatically slower than sklearn's sparse path. So when cuml is
    selected and ``X`` is sparse, we silently fall back to sklearn for
    that call — keeping cuml only for dense substrates (e.g. neural
    embeddings) where it actually wins.
    """
    global _SPARSE_FALLBACK_NOTICED
    backend, KMeansCls = _resolve_backend()
    if backend == "cuml" and hasattr(X, "toarray"):
        try:
            from selfchat.stats._kmeans_torch import fit_predict_sparse
            if not _SPARSE_FALLBACK_NOTICED:
                print("[selfchat] sparse input → torch sparse spherical-kmeans (GPU)")
                _SPARSE_FALLBACK_NOTICED = True
            return fit_predict_sparse(X, k=k, seed=seed)
        except ImportError:
            if not _SPARSE_FALLBACK_NOTICED:
                print("[selfchat] sparse input → sklearn (torch unavailable)")
                _SPARSE_FALLBACK_NOTICED = True
            from sklearn.cluster import KMeans as SkKMeans  # type: ignore[import-not-found]
            return SkKMeans(
                n_clusters=k, random_state=seed, n_init=n_init  # type: ignore[arg-type]
            ).fit_predict(X)
    if backend == "cuml":
        X = np.ascontiguousarray(X, dtype=np.float32)
        km = KMeansCls(
            n_clusters=k,
            random_state=seed,
            n_init=n_init,
            init="scalable-k-means++",
            max_iter=300,
        )
        km.fit(X)
        return np.asarray(km.labels_)
    return KMeansCls(
        n_clusters=k, random_state=seed, n_init=n_init  # type: ignore[arg-type]
    ).fit_predict(X)
