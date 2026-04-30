"""Clustering on per-message nomic embeddings — Phase 4.

Two methods on the same vectors:
  * k-means with k sweep over {2, 3, 5, 8, 10, 12, 15, 20, 30, 50}
  * HDBSCAN with min_cluster_size sweep over {10, 25, 50, 100, 200}

Reports per sweep step:
  * k-means: silhouette (subsampled), Davies-Bouldin, inertia
  * HDBSCAN: n_clusters, noise fraction, largest-cluster fraction

Method agreement: normalized mutual information between k-means at k* and
HDBSCAN at the min_cluster_size whose n_clusters is closest to k*.

Cluster characterization (at k*):
  * per-cluster n, fraction, turn_index mean/std, V/JB fractions, A/B
    fractions, top-3 seeds
  * MI(cluster, observable) normalized by H(observable) for turn_index
    (5-turn-bucketed), seed, variant, agent

Optional medoid + boundary message dump (--review out.txt) requires
re-reading transcripts to align text with embedding rows.

Run:
  python cluster.py
  python cluster.py --review cluster_review.txt
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from hdbscan import HDBSCAN  # type: ignore[import-not-found]
from sklearn.cluster import KMeans  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    davies_bouldin_score,
    mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)

K_SWEEP = (2, 3, 5, 8, 10, 12, 15, 20, 30, 50)
MCS_SWEEP = (10, 25, 50, 100, 200)
SILHOUETTE_SAMPLE = 3000
RNG_SEED = 0

logger = logging.getLogger(__name__)


# --- sweeps ----------------------------------------------------------------


def kmeans_sweep(
    X: np.ndarray, ks: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    """Fit KMeans for each k; return (labels_by_k, metrics_by_k)."""
    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    Xsamp = X[sample_idx]
    labels_by_k: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, float]] = {}
    print("=== k-means sweep ===")
    print(f"  {'k':>3}  {'silhouette':>10}  {'DB':>6}  {'inertia':>10}")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=10)
        labels = km.fit_predict(X)
        labels_by_k[k] = labels
        sil = float(silhouette_score(Xsamp, labels[sample_idx]))
        db = float(davies_bouldin_score(X, labels))
        inertia = float(km.inertia_)
        metrics[k] = {"silhouette": sil, "db": db, "inertia": inertia}
        print(f"  {k:>3}  {sil:>10.4f}  {db:>6.3f}  {inertia:>10.1f}")
    return labels_by_k, metrics


def hdbscan_sweep(
    X: np.ndarray, mcs_list: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    """Fit HDBSCAN for each min_cluster_size; return (labels_by_mcs, metrics_by_mcs)."""
    labels_by_mcs: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, float]] = {}
    print()
    print("=== HDBSCAN sweep ===")
    print(
        f"  {'mcs':>4}  {'n_clusters':>10}  {'noise_frac':>10}  "
        f"{'largest_frac':>12}"
    )
    for mcs in mcs_list:
        clu = HDBSCAN(min_cluster_size=mcs, min_samples=mcs, metric="euclidean")
        labels = clu.fit_predict(X)
        labels_by_mcs[mcs] = labels
        n_total = len(labels)
        n_noise = int((labels == -1).sum())
        clusters = [c for c in np.unique(labels) if c != -1]
        n_clusters = len(clusters)
        noise_frac = n_noise / n_total if n_total else 0.0
        largest = max(((labels == c).sum() for c in clusters), default=0)
        largest_frac = largest / n_total if n_total else 0.0
        metrics[mcs] = {
            "n_clusters": float(n_clusters),
            "noise_frac": noise_frac,
            "largest_frac": largest_frac,
        }
        print(
            f"  {mcs:>4}  {n_clusters:>10}  {noise_frac:>10.3f}  "
            f"{largest_frac:>12.3f}"
        )
    return labels_by_mcs, metrics


# --- selection -------------------------------------------------------------


def pick_kstar(metrics: dict[int, dict[str, float]]) -> int:
    """Pick k* — silhouette argmax (excluding k=2 trivial)."""
    sils = {k: m["silhouette"] for k, m in metrics.items() if k > 2}
    if not sils:
        return min(metrics)
    return max(sils, key=lambda k: sils[k])


def pick_hdbscan_primary(
    metrics: dict[int, dict[str, float]], target_n: int
) -> int:
    """Pick min_cluster_size whose n_clusters is closest to target_n,
    preferring settings with noise <= 80%."""
    candidates = [
        (mcs, m) for mcs, m in metrics.items() if m["noise_frac"] <= 0.80 and m["n_clusters"] >= 2
    ]
    if not candidates:
        candidates = list(metrics.items())
    return min(candidates, key=lambda kv: abs(kv[1]["n_clusters"] - target_n))[0]


# --- agreement -------------------------------------------------------------


def method_agreement(km_labels: np.ndarray, hdb_labels: np.ndarray) -> dict[str, float]:
    """NMI between two clusterings, considering only HDBSCAN non-noise points."""
    mask = hdb_labels != -1
    n_kept = int(mask.sum())
    if n_kept < 10:
        return {"nmi_all": float("nan"), "nmi_dense": float("nan"), "n_dense": float(n_kept)}
    nmi_all = float(normalized_mutual_info_score(km_labels, hdb_labels))
    nmi_dense = float(normalized_mutual_info_score(km_labels[mask], hdb_labels[mask]))
    return {"nmi_all": nmi_all, "nmi_dense": nmi_dense, "n_dense": float(n_kept)}


# --- characterization ------------------------------------------------------


def discretize_turn(turn_idx: np.ndarray, width: int = 5) -> np.ndarray:
    return (turn_idx // width).astype(np.int64)


def normalized_mi(labels: np.ndarray, observable: np.ndarray) -> float:
    """MI(labels; observable) / H(observable). Returns 0 if H(observable) ~= 0."""
    obs_int = _to_int_codes(observable)
    mi = float(mutual_info_score(labels, obs_int))
    counts = np.bincount(obs_int)
    p = counts / counts.sum()
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    if h <= 1e-12:
        return 0.0
    return mi / h


def _to_int_codes(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in ("i", "u"):
        # Re-pack to dense codes so bincount is well-defined for negative ids.
        unique, codes = np.unique(arr, return_inverse=True)
        del unique
        return codes.astype(np.int64)
    unique, codes = np.unique(arr, return_inverse=True)
    del unique
    return codes.astype(np.int64)


def per_cluster_summary(
    labels: np.ndarray,
    turn_idx: np.ndarray,
    variants: np.ndarray,
    agents: np.ndarray,
    seeds: np.ndarray,
) -> None:
    n_total = len(labels)
    cluster_ids = sorted(c for c in np.unique(labels) if c != -1)
    if not cluster_ids:
        print("  (no non-noise clusters)")
        return
    v_global = float((variants == "vanilla").mean())
    print()
    print(
        f"  {'id':>3}  {'n':>5}  {'frac':>5}  {'turn μ':>7}  {'turn σ':>6}  "
        f"{'V_frac':>6}  {'A_frac':>6}  {'top seed (frac)':<24}  "
        f"{'top2':<22}  {'top3':<22}"
    )
    for c in cluster_ids:
        mask = labels == c
        n = int(mask.sum())
        frac = n / n_total
        t = turn_idx[mask]
        tm = float(t.mean())
        ts = float(t.std())
        v_frac = float((variants[mask] == "vanilla").mean())
        a_frac = float((agents[mask] == "A").mean())
        seed_counts = Counter(seeds[mask].tolist())
        tops = seed_counts.most_common(3)
        cells = [f"{name} ({cnt / n:.0%})" for name, cnt in tops]
        while len(cells) < 3:
            cells.append("")
        print(
            f"  {c:>3}  {n:>5}  {frac:>5.2%}  {tm:>7.2f}  {ts:>6.2f}  "
            f"{v_frac:>6.2%}  {a_frac:>6.2%}  {cells[0]:<24}  "
            f"{cells[1]:<22}  {cells[2]:<22}"
        )
    print(f"  (global vanilla-fraction = {v_global:.2%})")


def observable_mi(
    labels: np.ndarray,
    turn_idx: np.ndarray,
    seeds: np.ndarray,
    variants: np.ndarray,
    agents: np.ndarray,
) -> None:
    print()
    print("  MI(cluster; observable) / H(observable):")
    items = [
        ("turn_index (5-bucket)", normalized_mi(labels, discretize_turn(turn_idx, 5))),
        ("seed", normalized_mi(labels, seeds)),
        ("variant", normalized_mi(labels, variants)),
        ("agent", normalized_mi(labels, agents)),
    ]
    items.sort(key=lambda kv: -kv[1])
    for name, val in items:
        print(f"    {name:<24} {val:>6.3f}")


# --- review dump -----------------------------------------------------------


def load_texts_aligned(
    transcript_dir: Path, run_ids: np.ndarray, turn_idx: np.ndarray
) -> list[str | None]:
    """Re-read transcripts to align message text to (run_id, turn_index) rows.

    Imported lazily because it requires the analyze module + transcript I/O,
    not needed for the metrics-only pass.
    """
    from analyze import load

    by_run: dict[str, dict[int, str]] = {}
    paths = sorted(transcript_dir.glob("*.jsonl"))
    needed = set(np.unique(run_ids).tolist())
    print(f"  scanning {len(paths)} transcript files for {len(needed)} run_ids...")
    for path in paths:
        t = load(path)
        if t is None or t.run_id not in needed:
            continue
        by_run[t.run_id] = {turn.turn_index: turn.content for turn in t.turns}
    out: list[str | None] = []
    missing = 0
    for rid, ti in zip(run_ids.tolist(), turn_idx.tolist()):
        text = by_run.get(rid, {}).get(int(ti))
        if text is None:
            missing += 1
        out.append(text)
    if missing:
        print(f"  WARNING: {missing} rows missing text")
    return out


def load_lengths_aligned(
    transcript_dir: Path, run_ids: np.ndarray, turn_idx: np.ndarray
) -> np.ndarray:
    """Re-read transcripts to compute char-length per row aligned to npz indices.

    Cheaper than load_texts_aligned because we only retain the int length, not
    the full message body. Missing rows get length 0 (will fail any positive
    --min-chars threshold).
    """
    texts = load_texts_aligned(transcript_dir, run_ids, turn_idx)
    return np.array([len(t) if t is not None else 0 for t in texts], dtype=np.int64)


def medoid_and_boundary_indices(
    X: np.ndarray, labels: np.ndarray, k_medoids: int = 5, k_boundary: int = 3
) -> dict[int, tuple[list[int], list[int]]]:
    """For each cluster, return (medoid_idxs, boundary_idxs).

    medoid: smallest distance to own centroid
    boundary: smallest |d_own - d_2nd_nearest| (most ambiguous)
    """
    cluster_ids = sorted(c for c in np.unique(labels) if c != -1)
    centroids = np.stack([X[labels == c].mean(axis=0) for c in cluster_ids])
    out: dict[int, tuple[list[int], list[int]]] = {}
    for ci, c in enumerate(cluster_ids):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        d_all = np.linalg.norm(X[idxs, None, :] - centroids[None, :, :], axis=2)
        d_own = d_all[:, ci]
        d_other = d_all.copy()
        d_other[:, ci] = np.inf
        d_2nd = d_other.min(axis=1)
        medoids = idxs[np.argsort(d_own)[:k_medoids]].tolist()
        ambig = np.abs(d_own - d_2nd)
        boundaries = idxs[np.argsort(ambig)[:k_boundary]].tolist()
        out[int(c)] = (medoids, boundaries)
    return out


def write_review(
    out_path: Path,
    medoids_boundaries: dict[int, tuple[list[int], list[int]]],
    texts: list[str | None],
    variants: np.ndarray,
    seeds: np.ndarray,
    agents: np.ndarray,
    turn_idx: np.ndarray,
    run_ids: np.ndarray,
    labels: np.ndarray,
) -> None:
    n_total = len(labels)
    with out_path.open("w") as f:
        f.write(f"# cluster review — n={n_total}\n\n")
        for c, (medoids, boundaries) in medoids_boundaries.items():
            n_c = int((labels == c).sum())
            f.write(f"## cluster {c}  (n={n_c}, frac={n_c / n_total:.1%})\n\n")
            f.write("### medoids (closest to cluster center)\n\n")
            for r in medoids:
                _write_row(f, r, texts, variants, seeds, agents, turn_idx, run_ids)
            f.write("\n### boundary (most ambiguous between this cluster and another)\n\n")
            for r in boundaries:
                _write_row(f, r, texts, variants, seeds, agents, turn_idx, run_ids)
            f.write("\n---\n\n")
    print(f"  → wrote review to {out_path}")


def _write_row(
    f,
    r: int,
    texts: list[str | None],
    variants: np.ndarray,
    seeds: np.ndarray,
    agents: np.ndarray,
    turn_idx: np.ndarray,
    run_ids: np.ndarray,
) -> None:
    text = texts[r] if r < len(texts) else None
    head = (
        f"- [row {r}] variant={variants[r]} seed={seeds[r]} "
        f"agent={agents[r]} turn={int(turn_idx[r])} run={run_ids[r][:8]}\n"
    )
    f.write(head)
    if text is None:
        f.write("    (text missing)\n\n")
        return
    snippet = text.strip()
    if len(snippet) > 1500:
        snippet = snippet[:1500] + " …[truncated]"
    indented = "\n".join("    " + line for line in snippet.splitlines())
    f.write(indented + "\n\n")


# --- main ------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?", default=Path("emb_msgs.npz"))
    p.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include degenerate-repetition runs (default: completed-only)",
    )
    p.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Write medoid+boundary text dump to this path (re-reads transcripts).",
    )
    p.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("archive/transcripts_nonspecific"),
        help="Where to find transcripts for --review (must match the source of the npz).",
    )
    p.add_argument("--k-override", type=int, default=None, help="Use this k* instead of auto-pick")
    p.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help=(
            "Drop messages shorter than this (re-reads transcripts to compute lengths). "
            "Use to exclude bracket-mode degenerate-style messages — distribution is "
            "bimodal at p25=24, p50=290, p75=2029, with the valley around 50–200. "
            "150 is a reasonable default cut."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not args.npz.exists():
        print(f"not found: {args.npz}\n\nrun `python embed_messages.py --out {args.npz}` first.")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    vecs = data["vectors"].astype(np.float32)
    turn_idx = data["turn_index"].astype(np.int64)
    variants = data["variants"].astype(str)
    seeds = data["seeds"].astype(str)
    run_ids = data["run_ids"].astype(str)
    stops = data["stop_reasons"].astype(str)
    agents = data["agents"].astype(str)

    print(f"loaded: n={len(vecs)} messages, dim={vecs.shape[1]}")
    print(f"  variants: {sorted(set(variants.tolist()))}")
    print(f"  seeds:    {sorted(set(seeds.tolist()))}")
    print(f"  stops:    {sorted(set(stops.tolist()))}")

    if not args.include_degenerate:
        keep = stops == "completed"
        n_drop = int((~keep).sum())
        if n_drop:
            print(
                f"\nfiltering to completed-only: dropped {n_drop} "
                "(use --include-degenerate to keep)"
            )
        vecs, turn_idx, variants, seeds, run_ids, stops, agents = (
            vecs[keep],
            turn_idx[keep],
            variants[keep],
            seeds[keep],
            run_ids[keep],
            stops[keep],
            agents[keep],
        )

    if args.min_chars > 0:
        print(f"\ncomputing message lengths from {args.transcript_dir} ...")
        lengths = load_lengths_aligned(args.transcript_dir, run_ids, turn_idx)
        keep = lengths >= args.min_chars
        n_drop = int((~keep).sum())
        print(
            f"--min-chars {args.min_chars}: dropped {n_drop} "
            f"({n_drop / len(lengths):.1%}), kept {int(keep.sum())}"
        )
        vecs, turn_idx, variants, seeds, run_ids, stops, agents = (
            vecs[keep],
            turn_idx[keep],
            variants[keep],
            seeds[keep],
            run_ids[keep],
            stops[keep],
            agents[keep],
        )

    print(f"  → using n={len(vecs)} messages\n")

    km_labels_by_k, km_metrics = kmeans_sweep(vecs, K_SWEEP)
    hdb_labels_by_mcs, hdb_metrics = hdbscan_sweep(vecs, MCS_SWEEP)

    if args.k_override is not None:
        kstar = args.k_override
        print(f"\n[--k-override] using k*={kstar}")
    else:
        kstar = pick_kstar(km_metrics)
        print(f"\nk* (silhouette argmax, k>2) = {kstar}")

    primary_mcs = pick_hdbscan_primary(hdb_metrics, target_n=kstar)
    print(
        f"HDBSCAN primary mcs = {primary_mcs} "
        f"(closest n_clusters={int(hdb_metrics[primary_mcs]['n_clusters'])} to k*={kstar})"
    )

    agree = method_agreement(km_labels_by_k[kstar], hdb_labels_by_mcs[primary_mcs])
    print(
        f"\nmethod agreement (k-means k={kstar} vs HDBSCAN mcs={primary_mcs}):"
    )
    print(
        f"  NMI all rows  = {agree['nmi_all']:.3f}    "
        f"NMI non-noise = {agree['nmi_dense']:.3f}    "
        f"n_dense = {int(agree['n_dense'])}"
    )

    print(f"\n=== k-means cluster characterization (k={kstar}) ===")
    per_cluster_summary(km_labels_by_k[kstar], turn_idx, variants, agents, seeds)
    observable_mi(km_labels_by_k[kstar], turn_idx, seeds, variants, agents)

    print(f"\n=== HDBSCAN cluster characterization (mcs={primary_mcs}) ===")
    hdb_labels = hdb_labels_by_mcs[primary_mcs]
    n_noise = int((hdb_labels == -1).sum())
    print(f"  noise: n={n_noise} ({n_noise / len(hdb_labels):.1%})")
    per_cluster_summary(hdb_labels, turn_idx, variants, agents, seeds)
    observable_mi(hdb_labels, turn_idx, seeds, variants, agents)

    if args.review is not None:
        print(f"\n=== writing review to {args.review} ===")
        texts = load_texts_aligned(args.transcript_dir, run_ids, turn_idx)
        mb = medoid_and_boundary_indices(vecs, km_labels_by_k[kstar])
        write_review(
            args.review.with_suffix(".kmeans.md"),
            mb,
            texts,
            variants,
            seeds,
            agents,
            turn_idx,
            run_ids,
            km_labels_by_k[kstar],
        )
        mb_hdb = medoid_and_boundary_indices(vecs, hdb_labels)
        write_review(
            args.review.with_suffix(".hdbscan.md"),
            mb_hdb,
            texts,
            variants,
            seeds,
            agents,
            turn_idx,
            run_ids,
            hdb_labels,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
