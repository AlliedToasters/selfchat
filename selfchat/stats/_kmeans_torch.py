"""GPU spherical k-means on sparse CSR via torch SpMM.

Avoids the densify-and-blow-up problem cuml has with high-dim TF-IDF.
The hot path is one ``X_csr @ C.T`` SpMM per iteration; centroid update
is ``(assignment_matrix @ X_csr).to_dense()`` followed by row L2-norm.

Inputs must already be row-L2-normalized (TfidfVectorizer with the
default ``norm='l2'`` satisfies this) — same precondition as sklearn
spherical-k-means.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.sparse import csr_matrix  # type: ignore[import-not-found]

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks")


def to_torch_csr(
    A: csr_matrix, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr).to(device, torch.int64),
        torch.from_numpy(A.indices).to(device, torch.int64),
        torch.from_numpy(A.data).to(device, dtype),
        size=A.shape,
    )


def normalize_sparse_rows_(X_csr: torch.Tensor) -> torch.Tensor:
    """In-place L2 row-normalize a torch CSR tensor."""
    indptr = X_csr.crow_indices()
    vals = X_csr.values()
    row_ids = torch.repeat_interleave(
        torch.arange(X_csr.shape[0], device=vals.device),
        indptr[1:] - indptr[:-1],
    )
    sq = torch.zeros(X_csr.shape[0], device=vals.device, dtype=vals.dtype)
    sq.index_add_(0, row_ids, vals * vals)
    norms = sq.clamp_min(1e-12).sqrt()
    vals.div_(norms[row_ids])
    return X_csr


@torch.no_grad()
def spherical_kmeans_sparse(
    X_csr: torch.Tensor, k: int, iters: int = 30, tol: float = 1e-4, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """X_csr: row-L2-normalized sparse CSR (N, D). Returns (labels, centroids)."""
    N, _ = X_csr.shape
    device = X_csr.values().device
    dtype = X_csr.values().dtype
    g = torch.Generator(device=device).manual_seed(seed)

    init_idx = torch.randperm(N, generator=g, device=device)[:k]
    sel = torch.sparse_coo_tensor(
        torch.stack([torch.arange(k, device=device), init_idx]),
        torch.ones(k, device=device, dtype=dtype),
        (k, N),
    ).to_sparse_csr()
    C = (sel @ X_csr).to_dense()
    C = torch.nn.functional.normalize(C, dim=1)

    labels = torch.zeros(N, dtype=torch.int64, device=device)
    for _ in range(iters):
        sims = X_csr @ C.T  # (N, k)
        labels = sims.argmax(1)

        M = torch.sparse_coo_tensor(
            torch.stack([labels, torch.arange(N, device=device)]),
            torch.ones(N, device=device, dtype=dtype),
            (k, N),
        ).to_sparse_csr()
        C_new = (M @ X_csr).to_dense()
        C_new = torch.nn.functional.normalize(C_new, dim=1)

        dead = C_new.abs().sum(1) == 0
        if dead.any():
            n_dead = int(dead.sum())
            reseed = torch.randperm(N, generator=g, device=device)[:n_dead]
            sel = torch.sparse_coo_tensor(
                torch.stack([torch.arange(n_dead, device=device), reseed]),
                torch.ones(n_dead, device=device, dtype=dtype),
                (n_dead, N),
            ).to_sparse_csr()
            C_new[dead] = torch.nn.functional.normalize(
                (sel @ X_csr).to_dense(), dim=1
            )

        delta = (C_new - C).norm().item()
        C = C_new
        if delta < tol:
            break

    return labels, C


def fit_predict_sparse(X: csr_matrix, k: int, seed: int, iters: int = 30) -> np.ndarray:
    """sklearn-style adapter: scipy CSR in, numpy labels out.

    Re-normalizes rows defensively (idempotent on already-L2-normalized).
    """
    X_csr = to_torch_csr(X, device="cuda", dtype=torch.float32)
    normalize_sparse_rows_(X_csr)
    labels, _ = spherical_kmeans_sparse(X_csr, k=k, iters=iters, seed=seed)
    return labels.cpu().numpy()
