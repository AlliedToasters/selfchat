"""Search for variant-pure clusters using SPHERICAL K-MEANS on word-level
TF-IDF features (posthoc, exploratory companion to `jb_purity_sweep.py`).

Motivation
----------
Phase 5 found word-TFIDF (1,2) outperforms 768-dim nomic embeddings as a
variant classifier (CV AUC ~0.86 on `freedom`, ~1.00 on `freedom_dark`,
versus T3 ~0.75 / ~0.997). If the JB-pure / V-pure attractor structure
reproduces from the lexical feature space, that's a replication of the
basin finding from a fully independent representation — and rules out
the basins being an embedding-specific artifact.

Method
------
* Same per-message substrate as `jb_purity_sweep`: messages from
  `artifacts/emb_msgs.npz` filtered to `stop_reason == "completed"` and
  `len(text) >= --min-chars`. We use the npz only for metadata alignment;
  the TF-IDF features are built directly from transcript text.
* Word-level TF-IDF (default ngram_range=(1, 2), `min_df=2`,
  `sublinear_tf=True`, project STOPWORDS, `lowercase=True`,
  `norm='l2'`). sklearn `TfidfVectorizer` L2-normalizes rows by default,
  so standard k-means on the resulting matrix implements spherical
  k-means (squared-Euclidean on unit vectors equals 2·(1 − cos sim)).
* Reuses the per-cluster purity machinery from `jb_purity_sweep`
  (`jb_purity_per_cluster`, K_SWEEP, size_floor) so results are
  apples-to-apples comparable to the embedding-based sweep.

Future work
-----------
NMF on the same TF-IDF matrix would give readable per-component
n-gram lists plus per-message loadings — a direct interpretability
layer for "what register characterizes the JB-affine basin" without
relying on medoid prose. Parked for now; this module is just the
spherical-k-means replication probe.

Run
---
  python -m selfchat.stats.lexical_purity_sweep --min-chars 150
  python -m selfchat.stats.lexical_purity_sweep --target vanilla \\
      --review notes/v_purity_lex.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.metrics.pairwise import euclidean_distances  # type: ignore[import-not-found]

from selfchat.core.shared import _STOPWORDS
from selfchat.stats._kmeans import fit_kmeans
from selfchat.stats.cluster import load_texts_aligned, write_review
from selfchat.stats.jb_purity_sweep import (
    K_SWEEP,
    SIZE_FLOOR_DEFAULT,
    jb_purity_per_cluster,
)

STOPWORDS = list(_STOPWORDS)


def fit_labels_spherical(X, k: int, seed: int) -> np.ndarray:
    """Spherical k-means via standard k-means on L2-normalized inputs."""
    return fit_kmeans(X, k, seed)


def medoid_and_boundary_indices_sparse(
    X, labels: np.ndarray, k_medoids: int = 8, k_boundary: int = 4
) -> dict[int, tuple[list[int], list[int]]]:
    """Sparse-aware medoid/boundary finder.

    For L2-normalized rows with cluster centroid c (||c|| ≤ 1), ranking
    by Euclidean distance to c is monotone in cosine similarity, so this
    returns the same medoid order as a cosine-medoid implementation.
    """
    cluster_ids = sorted(int(c) for c in np.unique(labels) if c != -1)
    centroids = np.vstack([
        np.asarray(X[labels == c].mean(axis=0)).ravel() for c in cluster_ids
    ])
    out: dict[int, tuple[list[int], list[int]]] = {}
    for ci, c in enumerate(cluster_ids):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        d_all = euclidean_distances(X[idxs], centroids)
        d_own = d_all[:, ci]
        d_other = d_all.copy()
        d_other[:, ci] = np.inf
        d_2nd = d_other.min(axis=1)
        medoids = idxs[np.argsort(d_own)[:k_medoids]].tolist()
        ambig = np.abs(d_own - d_2nd)
        boundaries = idxs[np.argsort(ambig)[:k_boundary]].tolist()
        out[c] = (medoids, boundaries)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?", default=Path("artifacts/emb_msgs.npz"),
                   help="Used for metadata alignment + completed-only filtering")
    p.add_argument("--transcript-dir", type=Path,
                   default=Path("archive/transcripts_nonspecific"))
    p.add_argument("--min-chars", type=int, default=150)
    p.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Random seeds for k-means. Multiple → stability report.")
    p.add_argument("--size-floor", type=int, default=SIZE_FLOOR_DEFAULT)
    p.add_argument("--target", choices=("jailbroken", "vanilla"), default="jailbroken",
                   help="Variant to rank for purity (default: jailbroken)")
    p.add_argument("--ngram-range", type=int, nargs=2, default=[1, 2])
    p.add_argument("--review", type=Path, default=None,
                   help="Dump medoid+boundary review of the global winner.")
    p.add_argument("--include-degenerate", action="store_true",
                   help="Include degenerate-repetition runs (default: completed-only)")
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    turn_idx = data["turn_index"].astype(np.int64)
    variants = data["variants"].astype(str)
    seeds_arr = data["seeds"].astype(str)
    run_ids = data["run_ids"].astype(str)
    stops = data["stop_reasons"].astype(str)
    agents = data["agents"].astype(str)

    print(f"loaded metadata: n={len(variants)}")

    if not args.include_degenerate:
        keep = stops == "completed"
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"completed-only: dropped {n_drop}")
        turn_idx, variants, seeds_arr, run_ids, stops, agents = (
            turn_idx[keep], variants[keep], seeds_arr[keep],
            run_ids[keep], stops[keep], agents[keep],
        )

    print(f"loading texts from {args.transcript_dir} ...")
    texts = load_texts_aligned(args.transcript_dir, run_ids, turn_idx)
    lengths = np.array([len(t) if t is not None else 0 for t in texts], dtype=np.int64)

    if args.min_chars > 0:
        keep = lengths >= args.min_chars
        n_drop = int((~keep).sum())
        print(
            f"--min-chars {args.min_chars}: dropped {n_drop} "
            f"({n_drop / len(lengths):.1%}), kept {int(keep.sum())}"
        )
        texts = [t for t, k in zip(texts, keep.tolist()) if k]
        turn_idx, variants, seeds_arr, run_ids, stops, agents = (
            turn_idx[keep], variants[keep], seeds_arr[keep],
            run_ids[keep], stops[keep], agents[keep],
        )

    texts_for_tfidf = [t if t is not None else "" for t in texts]

    ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))
    print(
        f"\nfitting TF-IDF: ngram_range={ngram_range}, min_df=2, "
        f"sublinear_tf=True, |stopwords|={len(STOPWORDS)}"
    )
    vec = TfidfVectorizer(
        ngram_range=ngram_range, min_df=2, lowercase=True,
        sublinear_tf=True, stop_words=STOPWORDS, norm="l2",
    )
    X = cast(csr_matrix, vec.fit_transform(texts_for_tfidf))
    n_rows, n_cols = cast(tuple[int, int], X.shape)
    density = X.nnz / (n_rows * n_cols) if n_cols else 0.0
    print(f"  matrix: shape=({n_rows}, {n_cols}), nnz={X.nnz}, density={density:.4f}")

    N = n_rows
    target = args.target
    target_label = "JB" if target == "jailbroken" else "V"
    M_target = int((variants == target).sum())
    base_target_frac = M_target / N if N else 0.0
    print(
        f"\nsubstrate: n={N}, "
        f"{target_label}_global={M_target} ({base_target_frac:.1%})"
    )
    print(
        f"max possible enrichment (cluster 100% {target_label}): "
        f"{1 / base_target_frac:.2f}"
    )

    total_clusters = sum(args.ks) * len(args.seeds)
    bonf = 0.05 / total_clusters if total_clusters else float("inf")
    print(
        f"\nsweep: ks={list(args.ks)}, seeds={args.seeds}, "
        f"size_floor={args.size_floor}, total_clusters={total_clusters}, "
        f"bonferroni α={bonf:.2e}"
    )

    # all_rows layout: (seed, k, cid, n, jb, jb_frac, enrichment, z, p, passes_bonf)
    all_rows: list[
        tuple[int, int, int, int, int, float, float, float, float, bool]
    ] = []
    labels_cache: dict[tuple[int, int], np.ndarray] = {}

    for seed in args.seeds:
        print(f"\n=== seed={seed} ===")
        print(
            f"  {'k':>4}  {'top_c':>5}  {'n':>5}  {target_label:>4}  "
            f"{f'{target_label}_frac':>7}  {'enrich':>6}  {'z':>6}  "
            f"{'p_hyper':>10}  {'sig':>4}"
        )
        for k in args.ks:
            labels = fit_labels_spherical(X, k, seed)
            labels_cache[(seed, k)] = labels
            rows = jb_purity_per_cluster(labels, variants, target=target)
            qualifying = [r for r in rows if r[1] >= args.size_floor]
            if not qualifying:
                print(f"  {k:>4}  (no cluster ≥ size_floor={args.size_floor})")
                continue
            top = max(qualifying, key=lambda r: (r[4], -r[6]))
            cid, n, jb, jb_frac, enrich, z, p = top
            sig = "***" if p < bonf else ("**" if p < 0.001 else ("*" if p < 0.05 else ""))
            for r in rows:
                all_rows.append((seed, k, *r, r[6] < bonf))
            print(
                f"  {k:>4}  {cid:>5}  {n:>5}  {jb:>4}  "
                f"{jb_frac:>7.1%}  {enrich:>6.2f}  {z:>6.2f}  "
                f"{p:>10.2e}  {sig:>4}"
            )

    if len(args.seeds) > 1:
        print(
            f"\n=== top {target_label}-enriched cluster per k, "
            f"across seeds (rank by enrichment) ==="
        )
        print(
            f"  {'k':>4}  {'best_enrich':>11}  {'best_n':>6}  "
            f"{'mean_enrich':>11}  {'std':>6}"
        )
        for k in args.ks:
            per_seed_top: list[tuple[float, int]] = []
            for seed in args.seeds:
                labels = labels_cache[(seed, k)]
                rows = jb_purity_per_cluster(labels, variants, target=target)
                qual = [r for r in rows if r[1] >= args.size_floor]
                if qual:
                    top = max(qual, key=lambda r: (r[4], -r[6]))
                    per_seed_top.append((top[4], top[1]))
            if not per_seed_top:
                continue
            enrichs = np.array([t[0] for t in per_seed_top])
            best_idx = int(enrichs.argmax())
            print(
                f"  {k:>4}  {per_seed_top[best_idx][0]:>11.2f}  "
                f"{per_seed_top[best_idx][1]:>6}  "
                f"{enrichs.mean():>11.2f}  {enrichs.std():>6.2f}"
            )

    print(f"\n=== global top {target_label}-enriched clusters (n >= size_floor) ===")
    print("    sort: enrichment desc; ties broken by smaller p_hyper")
    qualifying_all = [r for r in all_rows if r[3] >= args.size_floor]
    qualifying_all.sort(key=lambda r: (r[6], -r[8]), reverse=True)
    print(
        f"  {'seed':>4}  {'k':>4}  {'cid':>4}  {'n':>5}  {target_label:>4}  "
        f"{f'{target_label}_frac':>7}  {'enrich':>6}  {'z':>6}  "
        f"{'p_hyper':>10}  {'bonf':>5}"
    )
    for r in qualifying_all[:20]:
        seed, k, cid, n, jb, jb_frac, enrich, z, p, passes = r
        bonf_str = "yes" if passes else "no"
        print(
            f"  {seed:>4}  {k:>4}  {cid:>4}  {n:>5}  {jb:>4}  "
            f"{jb_frac:>7.1%}  {enrich:>6.2f}  {z:>6.2f}  "
            f"{p:>10.2e}  {bonf_str:>5}"
        )

    if not qualifying_all:
        print("  (none)")
        return 0

    if args.review is not None:
        winner = qualifying_all[0]
        seed, k, cid, n, jb, jb_frac, enrich, z, p, _ = winner
        print(
            f"\n=== writing medoid review of winner: seed={seed}, k={k}, "
            f"cid={cid}, n={n}, {target_label}_frac={jb_frac:.1%}, "
            f"enrich={enrich:.2f}, z={z:.2f}, p={p:.2e} ==="
        )
        labels = labels_cache[(seed, k)]
        mb_all = medoid_and_boundary_indices_sparse(
            X, labels, k_medoids=8, k_boundary=4
        )
        winner_only = {cid: mb_all[cid]} if cid in mb_all else {}
        write_review(
            args.review, winner_only, texts, variants, seeds_arr,
            agents, turn_idx, run_ids, labels,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
