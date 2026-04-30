"""PCA-by-turn-index figure over per-message embeddings.

Loads `emb_msgs.npz` from `embed_messages.py`, projects to 2D via PCA fit
on the full corpus, and renders a 1×2 facet (vanilla | jailbroken)
colored by turn_index. The visual question: do trajectories visibly arc
through PC space as conversations deepen?

Run: `python plot_msgs.py [emb_msgs.npz] [--out figures/pca_messages.png]`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2)), np.zeros(2)
    xc = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    proj = xc @ vt[:2].T
    total = float((s**2).sum()) or 1.0
    evr = (s[:2] ** 2) / total
    return proj, evr


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?", default=Path("emb_msgs.npz"))
    p.add_argument("--out", type=Path, default=Path("figures/pca_messages.png"))
    p.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include degenerate-repetition runs (default: completed-only)",
    )
    p.add_argument(
        "--seed",
        default="all",
        help="Restrict to a specific seed condition; default: all",
    )
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}\n\nrun `python embed_messages.py --out {args.npz}` first.")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    vecs = data["vectors"].astype(np.float32)
    turn_idx = data["turn_index"]
    variants = data["variants"].astype(str)
    seeds = data["seeds"].astype(str)
    stops = data["stop_reasons"].astype(str)

    if not args.include_degenerate:
        keep = stops == "completed"
        vecs, turn_idx, variants, seeds = vecs[keep], turn_idx[keep], variants[keep], seeds[keep]

    if args.seed != "all":
        keep = seeds == args.seed
        if not keep.any():
            print(f"seed {args.seed!r} not present after filtering.")
            return 1
        vecs, turn_idx, variants = vecs[keep], turn_idx[keep], variants[keep]

    if len(vecs) == 0:
        print("no messages to plot after filtering.")
        return 1

    proj, evr = pca_2d(vecs)
    print(f"PCA fit on n={len(vecs)}: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    vmin, vmax = int(turn_idx.min()), int(turn_idx.max())
    sc = None
    for ax, variant in zip(axes, ("vanilla", "jailbroken"), strict=True):
        mask = variants == variant
        if not mask.any():
            ax.set_title(f"{variant} (no data)")
            continue
        sc = ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=turn_idx[mask],
            cmap="viridis",
            s=6,
            alpha=0.55,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"{variant}  (n={int(mask.sum())})")
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)
        ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    axes[0].set_ylabel(f"PC2 ({evr[1]:.1%})")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("turn_index")
    suptitle = "per-message PCA, colored by turn depth"
    if args.seed != "all":
        suptitle += f"  (seed = {args.seed})"
    fig.suptitle(suptitle, y=0.99)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
