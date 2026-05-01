"""Search for JB-pure clusters across k-means scales (POSTHOC, exploratory).

Methodology:
  * Sweep k over a wide range on the length-filtered substrate.
  * For each (k, cluster), compute (n, jb_count, jb_frac) and a hypergeometric
    p-value vs the null "this cluster is a uniform random draw from the
    substrate" — small p means JB enrichment beyond size expectation.
  * Apply a size floor so we don't crown singletons.
  * Optionally rerun across multiple random seeds and report stability.
  * Optionally dump medoids of the top hit for manual review.

Bonferroni correction is applied across the total cluster count in the sweep.

Run:
  python jb_purity_sweep.py --min-chars 150
  python jb_purity_sweep.py --min-chars 150 --seeds 0 1 2 --review jb_purity.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom  # type: ignore[import-not-found]
from sklearn.cluster import KMeans  # type: ignore[import-not-found]

from selfchat.stats.cluster import (
    load_lengths_aligned,
    load_texts_aligned,
    medoid_and_boundary_indices,
    write_review,
)

K_SWEEP = (3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200)
SIZE_FLOOR_DEFAULT = 30


def jb_purity_per_cluster(
    labels: np.ndarray, variants: np.ndarray
) -> list[tuple[int, int, int, float, float, float, float]]:
    """For each cluster, return (cluster_id, n, jb_count, jb_frac, enrichment,
    z_score, p_hyper).

    enrichment = jb_frac / global_jb_frac  (1.0 = baseline; >1 = JB-enriched)
    z_score    = (jb_count - μ) / σ under hypergeometric null
                 (μ = n·M/N; σ from finite-population sampling). Calibrates
                 surprise jointly with sample size — complementary to enrichment,
                 which ignores size.
    p_hyper    = P(JB ≥ jb_count | N, M, n), one-sided upper tail.
    """
    N = len(labels)
    M = int((variants == "jailbroken").sum())
    base_jb_frac = M / N if N else 0.0
    out: list[tuple[int, int, int, float, float, float, float]] = []
    for c in sorted({int(c) for c in np.unique(labels) if c != -1}):
        mask = labels == c
        n = int(mask.sum())
        k = int((variants[mask] == "jailbroken").sum())
        jb_frac = k / n if n else 0.0
        enrichment = jb_frac / base_jb_frac if base_jb_frac > 0 else float("nan")
        if n > 0 and N > 1:
            mu = n * M / N
            var = n * M * (N - M) * (N - n) / (N * N * (N - 1)) if N > 1 else 0.0
            sigma = float(np.sqrt(var))
            z = (k - mu) / sigma if sigma > 0 else float("nan")
        else:
            z = float("nan")
        p = float(hypergeom.sf(k - 1, N, M, n)) if n else 1.0
        out.append((c, n, k, jb_frac, enrichment, z, p))
    return out


def fit_labels(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?", default=Path("artifacts/emb_msgs.npz"))
    p.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("archive/transcripts_nonspecific"),
    )
    p.add_argument("--min-chars", type=int, default=150)
    p.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds for k-means. Multiple → stability report.",
    )
    p.add_argument("--size-floor", type=int, default=SIZE_FLOOR_DEFAULT)
    p.add_argument(
        "--review",
        type=Path,
        default=None,
        help="If set, dump medoids+boundaries of the top JB-pure cluster (per k) "
        "to this path. Uses seeds[0] for the assignment.",
    )
    p.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include degenerate-repetition runs (default: completed-only)",
    )
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}\n\nrun `python embed_messages.py` first.")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    vecs = data["vectors"].astype(np.float32)
    turn_idx = data["turn_index"].astype(np.int64)
    variants = data["variants"].astype(str)
    seeds_arr = data["seeds"].astype(str)
    run_ids = data["run_ids"].astype(str)
    stops = data["stop_reasons"].astype(str)
    agents = data["agents"].astype(str)

    print(f"loaded: n={len(vecs)} messages, dim={vecs.shape[1]}")

    if not args.include_degenerate:
        keep = stops == "completed"
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"completed-only: dropped {n_drop}")
        vecs, turn_idx, variants, seeds_arr, run_ids, stops, agents = (
            vecs[keep],
            turn_idx[keep],
            variants[keep],
            seeds_arr[keep],
            run_ids[keep],
            stops[keep],
            agents[keep],
        )

    if args.min_chars > 0:
        print(f"computing message lengths from {args.transcript_dir} ...")
        lengths = load_lengths_aligned(args.transcript_dir, run_ids, turn_idx)
        keep = lengths >= args.min_chars
        n_drop = int((~keep).sum())
        print(
            f"--min-chars {args.min_chars}: dropped {n_drop} "
            f"({n_drop / len(lengths):.1%}), kept {int(keep.sum())}"
        )
        vecs, turn_idx, variants, seeds_arr, run_ids, stops, agents = (
            vecs[keep],
            turn_idx[keep],
            variants[keep],
            seeds_arr[keep],
            run_ids[keep],
            stops[keep],
            agents[keep],
        )

    N = len(vecs)
    M_jb = int((variants == "jailbroken").sum())
    base_jb = M_jb / N
    print(
        f"\nsubstrate: n={N}, JB_global={M_jb} ({base_jb:.1%}), "
        f"V_global={N - M_jb} ({1 - base_jb:.1%})"
    )

    total_clusters = sum(args.ks) * len(args.seeds)
    bonf = 0.05 / total_clusters if total_clusters else float("inf")
    print(
        f"\nsweep: ks={list(args.ks)}, seeds={args.seeds}, "
        f"size_floor={args.size_floor}, total_clusters={total_clusters}, "
        f"bonferroni α={bonf:.2e}"
    )

    # --- per-seed sweep --------------------------------------------------
    # all_rows layout: (seed, k, cid, n, jb, jb_frac, enrichment, z, p, passes_bonf)
    all_rows: list[
        tuple[int, int, int, int, int, float, float, float, float, bool]
    ] = []
    labels_cache: dict[tuple[int, int], np.ndarray] = {}

    for seed in args.seeds:
        print(f"\n=== seed={seed} ===")
        print(
            f"  {'k':>4}  {'top_c':>5}  {'n':>5}  {'JB':>4}  "
            f"{'JB_frac':>7}  {'enrich':>6}  {'z':>6}  {'p_hyper':>10}  {'sig':>4}"
        )
        for k in args.ks:
            labels = fit_labels(vecs, k, seed)
            labels_cache[(seed, k)] = labels
            rows = jb_purity_per_cluster(labels, variants)
            qualifying = [r for r in rows if r[1] >= args.size_floor]
            if not qualifying:
                print(f"  {k:>4}  (no cluster ≥ size_floor={args.size_floor})")
                continue
            # rank by enrichment desc, tiebreak on smaller p
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

    # --- best per k across seeds (for stability) -------------------------
    if len(args.seeds) > 1:
        print("\n=== top JB-enriched cluster per k, across seeds (rank by enrichment) ===")
        print(
            f"  {'k':>4}  {'best_enrich':>11}  {'best_n':>6}  "
            f"{'mean_enrich':>11}  {'std':>6}"
        )
        for k in args.ks:
            per_seed_top: list[tuple[float, int]] = []
            for seed in args.seeds:
                labels = labels_cache[(seed, k)]
                rows = jb_purity_per_cluster(labels, variants)
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

    # --- global winner across (seed, k) ---------------------------------
    print("\n=== global top JB-enriched clusters (n >= size_floor) ===")
    print("    sort: enrichment desc; ties broken by smaller p_hyper")
    qualifying_all = [r for r in all_rows if r[3] >= args.size_floor]
    qualifying_all.sort(key=lambda r: (r[6], -r[8]), reverse=True)
    print(
        f"  {'seed':>4}  {'k':>4}  {'cid':>4}  {'n':>5}  {'JB':>4}  "
        f"{'JB_frac':>7}  {'enrich':>6}  {'z':>6}  {'p_hyper':>10}  {'bonf':>5}"
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

    # --- review dump for the winner -------------------------------------
    if args.review is not None:
        winner = qualifying_all[0]
        seed, k, cid, n, jb, jb_frac, enrich, z, p, _ = winner
        print(
            f"\n=== writing medoid review of winner: seed={seed}, k={k}, cid={cid}, "
            f"n={n}, JB_frac={jb_frac:.1%}, enrich={enrich:.2f}, z={z:.2f}, "
            f"p={p:.2e} ==="
        )
        labels = labels_cache[(seed, k)]
        texts = load_texts_aligned(args.transcript_dir, run_ids, turn_idx)
        # dump only the winning cluster's medoids+boundaries (keep file small)
        mb_all = medoid_and_boundary_indices(vecs, labels, k_medoids=8, k_boundary=4)
        winner_only = {cid: mb_all[cid]} if cid in mb_all else {}
        write_review(
            args.review,
            winner_only,
            texts,
            variants,
            seeds_arr,
            agents,
            turn_idx,
            run_ids,
            labels,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
