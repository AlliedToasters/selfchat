"""POSTHOC: most JB-pure cluster from balanced-fit / full-assign kmeans.

Recipe (per user request, 2026-05-03):
  1. Load per-message neural embeddings, completed-only, len ≥ min_chars.
  2. Balance per (seed, variant) cell to N (default = smallest cell, 387
     on current substrate) → fit substrate.
  3. Fit kmeans on the balanced subsample.
  4. Re-assign the **full** (pre-balance) filtered dataset to clusters by
     nearest centroid.
  5. Report the most JB-pure cluster (with size floor) and characterize:
     prompt-seed distribution + turn_index (depth) distribution.

Why this differs from `jb_purity_sweep.py`: that one fits and measures on
the same imbalanced substrate, which manufactures purity in regions of
density imbalance (see `feedback_clustering_balance.md`). Here the
*partition geometry* is set by a balanced fit, so centroid placement is
not biased by class imbalance. Purity is then measured on the full data.

Note: this is post-hoc / exploratory. Predictions are not pre-registered;
this is descriptive characterisation following the manual-inspection
discussion.

Run:
  python -m selfchat.stats.jb_purity_balanced
  python -m selfchat.stats.jb_purity_balanced --ks 20 30 50 75
  python -m selfchat.stats.jb_purity_balanced --review notes/jb_purity_balanced.md
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom  # type: ignore[import-not-found]
from sklearn.cluster import KMeans  # type: ignore[import-not-found]

from selfchat.stats._kmeans import K_SWEEP, SIZE_FLOOR_DEFAULT
from selfchat.stats.cluster import (
    load_lengths_aligned,
    load_texts_aligned,
    write_review,
)
from selfchat.stats.purity_profile import balance_per_seed_variant


def fit_balanced_assign_full(
    X_full: np.ndarray, balanced_idx: np.ndarray, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans on X_full[balanced_idx]; predict labels on X_full.

    Returns (labels_full, centroids).
    """
    km = KMeans(
        n_clusters=k, random_state=seed, n_init=10  # type: ignore[arg-type]
    )
    km.fit(X_full[balanced_idx])
    labels_full = np.asarray(km.predict(X_full), dtype=np.int64)
    return labels_full, km.cluster_centers_


def jb_purity_table(
    labels: np.ndarray, variants: np.ndarray, target: str = "jailbroken"
) -> list[tuple[int, int, int, float, float, float, float]]:
    """Per cluster: (cid, n, target_count, target_frac, enrichment, z, p_hyper)."""
    N = len(labels)
    M = int((variants == target).sum())
    base = M / N if N else 0.0
    out: list[tuple[int, int, int, float, float, float, float]] = []
    for c in sorted({int(c) for c in np.unique(labels) if c != -1}):
        mask = labels == c
        n = int(mask.sum())
        k = int((variants[mask] == target).sum())
        frac = k / n if n else 0.0
        enrichment = frac / base if base > 0 else float("nan")
        if n > 0 and N > 1:
            mu = n * M / N
            var = n * M * (N - M) * (N - n) / (N * N * (N - 1))
            sigma = float(np.sqrt(var))
            z = (k - mu) / sigma if sigma > 0 else float("nan")
        else:
            z = float("nan")
        p = float(hypergeom.sf(k - 1, N, M, n)) if n else 1.0
        out.append((c, n, k, frac, enrichment, z, p))
    return out


def characterize_cluster(
    cid: int,
    labels: np.ndarray,
    variants: np.ndarray,
    seeds_arr: np.ndarray,
    turn_idx: np.ndarray,
    agents: np.ndarray,
    run_ids: np.ndarray,
) -> None:
    mask = labels == cid
    n = int(mask.sum())
    if n == 0:
        print(f"  cluster {cid}: empty")
        return

    v_in = variants[mask]
    s_in = seeds_arr[mask]
    t_in = turn_idx[mask]
    a_in = agents[mask]
    r_in = run_ids[mask]

    n_jb = int((v_in == "jailbroken").sum())
    n_v = int((v_in == "vanilla").sum())

    print(f"\n--- cluster {cid} ---")
    print(f"  size: n={n} ({n / len(labels):.2%} of full assigned data)")
    print(f"  variant split: JB={n_jb} ({n_jb/n:.1%})  V={n_v} ({n_v/n:.1%})")
    print(f"  agent split: A={int((a_in == 'A').sum())} "
          f"B={int((a_in == 'B').sum())}")
    print(f"  unique runs: {len(np.unique(r_in))}")

    print()
    print("  prompt-seed distribution (cluster):")
    seed_counts = Counter(s_in.tolist())
    seed_global = Counter(seeds_arr.tolist())
    print(f"    {'seed':<24} {'n':>5} {'cluster%':>8} {'global%':>7} "
          f"{'over-rep':>8}")
    for seed_name, cnt in seed_counts.most_common():
        cluster_frac = cnt / n
        global_frac = seed_global[seed_name] / len(labels)
        overrep = cluster_frac / global_frac if global_frac > 0 else float("nan")
        print(f"    {seed_name:<24} {cnt:>5} {cluster_frac:>8.1%} "
              f"{global_frac:>7.1%} {overrep:>8.2f}x")

    print()
    print("  prompt-seed × variant (cluster):")
    print(f"    {'seed':<24} {'JB':>5} {'V':>5} {'JB%':>7}")
    for seed_name in sorted(seed_counts.keys()):
        seed_mask = s_in == seed_name
        jb_n = int(((v_in == "jailbroken") & seed_mask).sum())
        v_n = int(((v_in == "vanilla") & seed_mask).sum())
        tot = jb_n + v_n
        jb_frac = jb_n / tot if tot else 0.0
        print(f"    {seed_name:<24} {jb_n:>5} {v_n:>5} {jb_frac:>7.1%}")

    print()
    print("  turn_index (depth) distribution:")
    pcts = np.percentile(t_in, [5, 25, 50, 75, 95])
    print(f"    n={n}  μ={t_in.mean():.2f}  σ={t_in.std():.2f}  "
          f"min={int(t_in.min())}  max={int(t_in.max())}")
    print(f"    p5={pcts[0]:.1f}  p25={pcts[1]:.1f}  p50={pcts[2]:.1f}  "
          f"p75={pcts[3]:.1f}  p95={pcts[4]:.1f}")

    print()
    print("  turn_index histogram (5-turn buckets):")
    edges = np.arange(0, max(int(t_in.max()) + 6, 51), 5)
    hist, _ = np.histogram(t_in, bins=edges)
    bar_max = int(hist.max()) if hist.size else 1
    for i, cnt in enumerate(hist.tolist()):
        if cnt == 0 and i > len(hist) - 3:
            continue
        bar = "█" * int(40 * cnt / bar_max) if bar_max else ""
        print(f"    [{int(edges[i]):>3}-{int(edges[i+1])-1:>3}] "
              f"{cnt:>5}  {bar}")

    # Compare cluster depth to global depth on this same substrate
    print()
    print("  vs. full-substrate turn_index:")
    print(f"    cluster μ={t_in.mean():.2f}  global μ={turn_idx.mean():.2f}  "
          f"Δ={t_in.mean() - turn_idx.mean():+.2f}")
    print(f"    cluster median={np.median(t_in):.1f}  "
          f"global median={np.median(turn_idx):.1f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?",
                   default=Path("artifacts/emb_msgs.npz"))
    p.add_argument("--transcript-dir", type=Path,
                   default=Path("transcripts/"))
    p.add_argument("--min-chars", type=int, default=150)
    p.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    p.add_argument("--kmeans-seed", type=int, default=0,
                   help="Single random seed for kmeans (this is a "
                        "characterization run, not a stability sweep).")
    p.add_argument("--balance-target", type=int, default=None,
                   help="Per-cell sample target (default: smallest cell).")
    p.add_argument("--size-floor", type=int, default=SIZE_FLOOR_DEFAULT)
    p.add_argument("--include-degenerate", action="store_true")
    p.add_argument("--review", type=Path, default=None,
                   help="Dump medoids+boundaries of the global winner cluster.")
    p.add_argument("--top-n-each-k", type=int, default=1,
                   help="Show top-N JB-enriched clusters per k.")
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}")
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
        print(f"completed-only: dropped {n_drop}")
        vecs, turn_idx, variants, seeds_arr, run_ids, stops, agents = (
            vecs[keep], turn_idx[keep], variants[keep], seeds_arr[keep],
            run_ids[keep], stops[keep], agents[keep],
        )

    if args.min_chars > 0:
        print(f"computing message lengths from {args.transcript_dir} ...")
        lengths = load_lengths_aligned(args.transcript_dir, run_ids, turn_idx)
        keep = lengths >= args.min_chars
        n_drop = int((~keep).sum())
        print(f"--min-chars {args.min_chars}: dropped {n_drop} "
              f"({n_drop / len(lengths):.1%}), kept {int(keep.sum())}")
        vecs, turn_idx, variants, seeds_arr, run_ids, stops, agents = (
            vecs[keep], turn_idx[keep], variants[keep], seeds_arr[keep],
            run_ids[keep], stops[keep], agents[keep],
        )

    N_full = len(vecs)
    base_jb_full = (variants == "jailbroken").mean()
    print(f"\nfull (pre-balance) substrate: n={N_full}, "
          f"JB={int((variants == 'jailbroken').sum())} ({base_jb_full:.1%})")

    # Per-cell breakdown.
    cell_counts = Counter(zip(seeds_arr.tolist(), variants.tolist()))
    print(f"  per (seed, variant) cell sizes (min={min(cell_counts.values())}):")
    for (s, v), c in sorted(cell_counts.items()):
        print(f"    {s:<24}/{('JB' if v == 'jailbroken' else 'V'):<2}  {c:>5}")

    # Balance per (seed, variant) cell.
    target_n = args.balance_target or min(cell_counts.values())
    balanced_idx = balance_per_seed_variant(
        seeds_arr, variants, target_n, args.kmeans_seed
    )
    print(f"\nbalanced fit-substrate: target_n={target_n} per cell, "
          f"n={len(balanced_idx)}")
    bal_cells = Counter(
        zip(seeds_arr[balanced_idx].tolist(), variants[balanced_idx].tolist())
    )
    bal_jb = (variants[balanced_idx] == "jailbroken").mean()
    print(f"  balanced JB fraction = {bal_jb:.1%} (should be ~50%)")
    assert all(c == target_n for c in bal_cells.values()), \
        f"balancing failed; cells: {bal_cells}"

    # Sweep k.
    print(f"\n=== sweep ks={list(args.ks)} (kmeans_seed={args.kmeans_seed}) ===")
    print(f"  {'k':>4}  {'cid':>4}  {'n':>5}  {'JB':>5}  {'JB%':>6}  "
          f"{'enrich':>6}  {'z':>6}  {'p_hyper':>10}")
    all_winners: list[tuple[int, int, int, int, int, float, float, float, float]] = []
    labels_cache: dict[int, np.ndarray] = {}
    for k in args.ks:
        labels_full, _ = fit_balanced_assign_full(
            vecs, balanced_idx, k, args.kmeans_seed
        )
        labels_cache[k] = labels_full
        rows = jb_purity_table(labels_full, variants, target="jailbroken")
        qual = [r for r in rows if r[1] >= args.size_floor]
        if not qual:
            print(f"  {k:>4}  (no cluster ≥ size_floor={args.size_floor})")
            continue
        # rank by enrichment desc; tiebreak smaller p
        ranked = sorted(qual, key=lambda r: (r[4], -r[6]), reverse=True)
        for r in ranked[: args.top_n_each_k]:
            cid, n, jb, frac, enrich, z, ph = r
            print(f"  {k:>4}  {cid:>4}  {n:>5}  {jb:>5}  "
                  f"{frac:>6.1%}  {enrich:>6.2f}  {z:>6.2f}  {ph:>10.2e}")
            all_winners.append((k, cid, n, jb, int(n - jb), frac, enrich, z, ph))

    if not all_winners:
        print("\nno qualifying clusters found")
        return 0

    # Pick global winner: highest JB fraction, tiebreak by enrichment.
    # On full assigned data, JB-fraction = enrichment × base_jb_full, so
    # enrichment ranking is equivalent — use the explicit fraction for clarity.
    all_winners.sort(key=lambda r: (r[5], r[6]), reverse=True)
    winner = all_winners[0]
    k_w, cid_w, n_w, jb_w, _, frac_w, enr_w, z_w, p_w = winner
    print()
    print("=" * 72)
    print(
        f"GLOBAL WINNER: k={k_w}, cid={cid_w}, n={n_w}, JB={jb_w}/{n_w} "
        f"({frac_w:.1%}), enrich={enr_w:.2f}, z={z_w:.2f}, p={p_w:.2e}"
    )
    print("=" * 72)

    characterize_cluster(
        cid_w, labels_cache[k_w], variants, seeds_arr, turn_idx, agents, run_ids
    )

    if args.review is not None:
        from selfchat.stats.cluster import medoid_and_boundary_indices
        print(f"\n=== writing medoid review of winner to {args.review} ===")
        texts = load_texts_aligned(args.transcript_dir, run_ids, turn_idx)
        labels_full = labels_cache[k_w]
        mb_all = medoid_and_boundary_indices(
            vecs, labels_full, k_medoids=8, k_boundary=4
        )
        winner_only = {cid_w: mb_all[cid_w]} if cid_w in mb_all else {}
        write_review(
            args.review, winner_only, texts, variants, seeds_arr, agents,
            turn_idx, run_ids, labels_full,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
