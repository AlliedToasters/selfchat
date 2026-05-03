"""Consolidated purity sweep — variants × prompt-seeds × feature-spaces.

For both neural-embedding and lexical-TFIDF feature spaces, sweeps
k-means across `K_SWEEP` and computes:

* max-purity-per-variant: max over clusters with ``n >= size_floor`` of
  the variant's fraction in the cluster. Two values per k (V, JB).
* max-purity-per-prompt-seed: max over clusters of the prompt-seed's
  fraction in the cluster. n_seeds values per k.

Substrate is balanced by deterministic per-seed subsample to equal
counts (default: subsample down to the smallest seed's count) before
clustering. This keeps the seed-purity baseline uniform across seeds —
without balancing, a more-frequent seed has a higher random-baseline
ceiling regardless of attractor structure. The variant panel is
unaffected because V/JB is ~50/50 by construction.

Output JSON cache at ``artifacts/purity_profile.json`` consumed by
``selfchat.viz.plot_purity_profile``.

Run:
  python -m selfchat.stats.purity_profile
  python -m selfchat.stats.purity_profile --no-balance      # raw substrate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]

from selfchat.core.shared import _STOPWORDS
from selfchat.stats._kmeans import fit_kmeans, K_SWEEP, SIZE_FLOOR_DEFAULT
from selfchat.stats.cluster import load_texts_aligned

STOPWORDS = list(_STOPWORDS)

def balance_per_seed(
    seeds: np.ndarray, target_n: int | None, rng_seed: int
) -> np.ndarray:
    """Return sorted indices into `seeds` that subsample to `target_n` per seed.

    If `target_n` is None, use min(count_per_seed) — all seeds matched
    to the smallest seed's count.
    """
    counts = Counter(seeds.tolist())
    if target_n is None:
        target_n = min(counts.values())
    rng = np.random.default_rng(rng_seed)
    keep_idx: list[int] = []
    for s in sorted(counts.keys()):
        seed_idx = np.where(seeds == s)[0]
        if len(seed_idx) <= target_n:
            keep_idx.extend(seed_idx.tolist())
        else:
            sub = rng.choice(seed_idx, size=target_n, replace=False)
            keep_idx.extend(sub.tolist())
    return np.array(sorted(keep_idx))


def max_target_purity(
    labels: np.ndarray, target_mask: np.ndarray, size_floor: int
) -> float:
    """Max fraction of `target_mask=True` over clusters with n >= size_floor.

    Returns NaN if no cluster meets the size floor.
    """
    best = float("nan")
    for c in np.unique(labels):
        if c == -1:
            continue
        mask = labels == c
        n = int(mask.sum())
        if n < size_floor:
            continue
        frac = float(target_mask[mask].mean())
        if np.isnan(best) or frac > best:
            best = frac
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?",
                   default=Path("artifacts/emb_msgs.npz"))
    p.add_argument("--transcript-dir", type=Path,
                   default=Path("transcripts/"))
    p.add_argument("--min-chars", type=int, default=150)
    p.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    p.add_argument("--kmeans-seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="K-means random seeds (mean ± std reported across them).")
    p.add_argument("--size-floor", type=int, default=SIZE_FLOOR_DEFAULT)
    p.add_argument("--ngram-range", type=int, nargs=2, default=[1, 2])
    p.add_argument("--no-balance", action="store_true",
                   help="Skip the balanced-subsample step (use raw substrate).")
    p.add_argument("--balance-target", type=int, default=None,
                   help="Override per-seed sample target (default: min count).")
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/purity_profile.json"))
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

    print(f"loaded: n={len(vecs)}, dim={vecs.shape[1]}")

    keep = stops == "completed"
    print(f"completed-only: kept {int(keep.sum())} / {len(keep)}")
    vecs, turn_idx, variants, seeds_arr, run_ids = (
        vecs[keep], turn_idx[keep], variants[keep], seeds_arr[keep], run_ids[keep]
    )

    print(f"loading texts from {args.transcript_dir} ...")
    texts = load_texts_aligned(args.transcript_dir, run_ids, turn_idx)
    lengths = np.array([len(t) if t is not None else 0 for t in texts], dtype=np.int64)

    keep = lengths >= args.min_chars
    n_drop = int((~keep).sum())
    print(
        f"--min-chars {args.min_chars}: dropped {n_drop} "
        f"({n_drop / len(lengths):.1%}), kept {int(keep.sum())}"
    )
    texts = [t for t, kk in zip(texts, keep.tolist()) if kk]
    vecs, turn_idx, variants, seeds_arr, run_ids = (
        vecs[keep], turn_idx[keep], variants[keep], seeds_arr[keep], run_ids[keep]
    )

    if not args.no_balance:
        before = Counter(seeds_arr.tolist())
        keep_idx = balance_per_seed(seeds_arr, args.balance_target, args.kmeans_seeds[0])
        n_before = len(seeds_arr)
        vecs = vecs[keep_idx]
        turn_idx = turn_idx[keep_idx]
        variants = variants[keep_idx]
        seeds_arr = seeds_arr[keep_idx]
        run_ids = run_ids[keep_idx]
        texts = [texts[i] for i in keep_idx.tolist()]
        after = Counter(seeds_arr.tolist())
        target_n = (
            min(before.values()) if args.balance_target is None else args.balance_target
        )
        print(
            f"\nbalance per-seed → target_n={target_n}: "
            f"n_before={n_before}, n_after={len(seeds_arr)}"
        )
        for s in sorted(before.keys()):
            print(f"  {s:>24}  {before[s]:>5} → {after.get(s, 0):>5}")
    else:
        print("(no-balance) using raw substrate")

    texts_clean = [t if t is not None else "" for t in texts]

    ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))
    print(f"\nfitting TF-IDF: ngram_range={ngram_range} ...")
    vec = TfidfVectorizer(
        ngram_range=ngram_range, min_df=2, lowercase=True,
        sublinear_tf=True, stop_words=STOPWORDS, norm="l2",
    )
    X_lex = cast(csr_matrix, vec.fit_transform(texts_clean))
    n_rows, n_cols = cast(tuple[int, int], X_lex.shape)
    print(f"  TF-IDF shape: ({n_rows}, {n_cols}), nnz={X_lex.nnz}")

    seeds_present = sorted(set(seeds_arr.tolist()))
    print(f"\nseeds present: {seeds_present}")

    # Per (space, target, k) collect one purity value per kmeans seed,
    # then aggregate to mean ± std.
    raw_variant: dict[str, dict[str, list[list[float]]]] = {
        "neural": {"jailbroken": [], "vanilla": []},
        "lexical": {"jailbroken": [], "vanilla": []},
    }
    raw_seed: dict[str, dict[str, list[list[float]]]] = {
        "neural": {s: [] for s in seeds_present},
        "lexical": {s: [] for s in seeds_present},
    }

    n_kseeds = len(args.kmeans_seeds)
    for space, X in (("neural", vecs), ("lexical", X_lex)):
        print(f"\n=== {space} sweep (kmeans_seeds={args.kmeans_seeds}) ===")
        header_seeds = "  ".join(f"{s[:10]:>10}" for s in seeds_present)
        print(f"  {'k':>4}  {'V (μ±σ)':>13}  {'JB (μ±σ)':>13}  | {header_seeds}")
        for k in args.ks:
            v_runs: list[float] = []
            j_runs: list[float] = []
            seed_runs: dict[str, list[float]] = {s: [] for s in seeds_present}
            for ks in args.kmeans_seeds:
                labels = fit_kmeans(X, k, ks)
                v_runs.append(max_target_purity(
                    labels, variants == "vanilla", args.size_floor
                ))
                j_runs.append(max_target_purity(
                    labels, variants == "jailbroken", args.size_floor
                ))
                for s in seeds_present:
                    seed_runs[s].append(max_target_purity(
                        labels, seeds_arr == s, args.size_floor
                    ))
            raw_variant[space]["vanilla"].append(v_runs)
            raw_variant[space]["jailbroken"].append(j_runs)
            for s in seeds_present:
                raw_seed[space][s].append(seed_runs[s])
            v_arr = np.array(v_runs)
            j_arr = np.array(j_runs)
            seed_str = "  ".join(
                f"{np.mean(seed_runs[s]):>10.3f}" for s in seeds_present
            )
            print(
                f"  {k:>4}  {v_arr.mean():>5.3f}±{v_arr.std():.3f}  "
                f"{j_arr.mean():>5.3f}±{j_arr.std():.3f}  | {seed_str}"
            )

    def aggregate(
        raw: dict[str, dict[str, list[list[float]]]],
    ) -> dict[str, dict[str, dict[str, list[float]]]]:
        out: dict[str, dict[str, dict[str, list[float]]]] = {}
        for space, by_target in raw.items():
            out[space] = {}
            for target, runs_per_k in by_target.items():
                arr = np.array(runs_per_k, dtype=float)  # shape (n_k, n_seeds)
                out[space][target] = {
                    "mean": np.nanmean(arr, axis=1).tolist(),
                    "std": np.nanstd(arr, axis=1).tolist(),
                }
        return out

    out = {
        "metadata": {
            "n": int(len(seeds_arr)),
            "min_chars": int(args.min_chars),
            "kmeans_seeds": list(args.kmeans_seeds),
            "n_kmeans_seeds": n_kseeds,
            "size_floor": int(args.size_floor),
            "ks": list(args.ks),
            "seeds_present": seeds_present,
            "balanced": (not args.no_balance),
            "balance_target": (
                None if args.balance_target is None else int(args.balance_target)
            ),
            "ngram_range": list(ngram_range),
        },
        "by_variant": aggregate(raw_variant),
        "by_seed": aggregate(raw_seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
