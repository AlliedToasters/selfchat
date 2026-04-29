"""per-seed marginal separability tests on PC1 / PC2.

For each seed condition in the dataset, runs Mann-Whitney U comparing
vanilla vs jailbroken on the PCA-projected coordinates. Non-parametric
(no normality assumption), appropriate for small-N cells.

PCA is fit on the FULL corpus so the projection axes are shared across
conditions; per-seed tests then ask "given the corpus-wide axes of
variation, do the two variants occupy different positions within this
seed?" Stratifying by seed avoids confounding seed-level variance with
variant-level variance — pooling across seeds would mix the two.

Run: `python separability.py emb.npz`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from analyze import is_current_seed


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection (N, 2), explained_variance_ratio (2,)). Inlined to
    avoid pulling in matplotlib via plot.py."""
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2)), np.zeros(2)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    proj = xc @ vt[:2].T
    total_var = float((s**2).sum()) or 1.0
    evr = (s[:2] ** 2) / total_var
    return proj, evr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="Path to .npz from embed.py")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for the marker column (default: 0.05).",
    )
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    vecs = data["vectors"]
    variants = data["variants"].astype(str)
    seeds = data["seeds"].astype(str)

    proj, evr = pca_2d(vecs)
    n_total = len(vecs)
    print(f"PCA fit on n={n_total}: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}")
    print()

    seed_names = [s for s in sorted(set(seeds.tolist())) if is_current_seed(s)]
    n_tests = 2 * len(seed_names)
    bonferroni = args.alpha / n_tests if n_tests else float("nan")

    def mark(p: float) -> str:
        if p < bonferroni:
            return "**"
        if p < args.alpha:
            return " *"
        return "  "

    def row(label: str, mask: np.ndarray) -> None:
        van = mask & (variants == "vanilla")
        jb = mask & (variants == "jailbroken")
        n_van = int(van.sum())
        n_jb = int(jb.sum())
        if n_van < 2 or n_jb < 2:
            print(f"{label:<10}  {n_van:>5}  {n_jb:>5}  (insufficient samples)")
            return
        cells: dict[str, tuple[float, float]] = {}
        for pc_name, pc_idx in (("PC1", 0), ("PC2", 1)):
            _u, p_val = stats.mannwhitneyu(
                proj[van, pc_idx], proj[jb, pc_idx], alternative="two-sided"
            )
            d_median = float(np.median(proj[jb, pc_idx]) - np.median(proj[van, pc_idx]))
            cells[pc_name] = (d_median, float(p_val))
        print(
            f"{label:<10}  {n_van:>5}  {n_jb:>5}  "
            f"{cells['PC1'][0]:>+10.3f}  {cells['PC1'][1]:>7.4f}{mark(cells['PC1'][1])}  "
            f"{cells['PC2'][0]:>+10.3f}  {cells['PC2'][1]:>7.4f}{mark(cells['PC2'][1])}"
        )

    cols = (
        f"{'seed':<10}  {'n_van':>5}  {'n_jb':>5}  "
        f"{'PC1 Δmed':>10}  {'PC1 p':>9}  "
        f"{'PC2 Δmed':>10}  {'PC2 p':>9}"
    )
    print(cols)
    print("-" * len(cols))

    for seed in seed_names:
        row(seed, seeds == seed)

    print("-" * len(cols))
    row("all", np.isin(seeds, seed_names))

    print()
    print("  Δmed   = median(jailbroken) - median(vanilla)  [in PC-projection units]")
    print("           positive = jailbroken shifted higher on that PC")
    print("  p      = Mann-Whitney U two-sided")
    print(f"  *      = p < α = {args.alpha}")
    if n_tests:
        print(
            f"  **     = p < α/{n_tests} = {bonferroni:.4f}  "
            f"(Bonferroni-corrected for the {n_tests} per-seed tests)"
        )
    print()
    print("  note: the 'all' row pools across seeds. It tests a different question than the")
    print("        per-seed rows ('do variants differ ignoring condition?' vs. 'do variants")
    print("        differ within this condition?'), so its p-value is not Bonferroni-corrected")
    print("        with the per-seed family. Pooling can also surface false positives if one")
    print("        seed dominates the corpus or if seeds are imbalanced between variants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
