"""2D PCA scatter of terminal-state embeddings.

Loads a .npz produced by `embed.py`, projects vectors to 2D via numpy SVD,
and renders a scatter colored by cell (variant×seed) with marker shape per
stop_reason. Saves PNG (and optionally PDF). No interactive display.

Run: `python plot.py emb.npz [--out figures/pca_terminal.png]`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (projection (N,2), explained_variance_ratio (2,)).

    Pure numpy SVD; no scikit-learn needed.
    """
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
        "--out",
        type=Path,
        default=Path("figures/pca_terminal.png"),
        help="Output PNG path (default: figures/pca_terminal.png)",
    )
    p.add_argument("--title", default=None)
    p.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Plot only these seeds (default: all). PCA is still fit on all data.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Plot only these variants (default: all). PCA is still fit on all data.",
    )
    p.add_argument(
        "--stops",
        nargs="+",
        default=None,
        help="Plot only these stop reasons (default: all). PCA is still fit on all data.",
    )
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    vecs = data["vectors"]
    variants = data["variants"]
    seeds = data["seeds"]
    stop_reasons = data["stop_reasons"]
    last_k = int(data["last_k"]) if "last_k" in data.files else None
    model = str(data["model"]) if "model" in data.files else "unknown"

    print(f"loaded: {vecs.shape[0]} vectors, dim={vecs.shape[1]}, model={model}")

    proj, evr = pca_2d(vecs)
    print(f"PCA explained variance: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}  (fit on n={len(vecs)})")

    # Filter for plotting AFTER PCA fit so the projection reflects the full
    # variance structure even when only a subset is shown.
    n_fit = len(vecs)
    mask = np.ones(n_fit, dtype=bool)
    if args.seeds:
        mask &= np.isin(seeds, args.seeds)
    if args.variants:
        mask &= np.isin(variants, args.variants)
    if args.stops:
        mask &= np.isin(stop_reasons, args.stops)

    n_shown = int(mask.sum())
    if n_shown == 0:
        print("filter excluded every point; nothing to plot")
        return 1
    if n_shown < n_fit:
        print(f"filter: showing {n_shown}/{n_fit} points")

    df = pd.DataFrame(
        {
            "PC1": proj[mask, 0],
            "PC2": proj[mask, 1],
            "cell": [f"{v}|{s}" for v, s in zip(variants[mask], seeds[mask], strict=True)],
            "stop": stop_reasons[mask],
        }
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 7))

    # Color encoding: variant -> hue family, seed -> lightness within family.
    # Variants are the salient signal (the experimental contrast), so they get
    # the most distinguishable channel (hue). Seeds modulate within each.
    # jailbroken = red ("red team" convention); vanilla = blue.
    HUE_FAMILIES = {"vanilla": "Blues", "jailbroken": "Reds"}
    SEED_VISUAL_ORDER = [  # mild -> adversarial
        "task",
        "freedom",
        "alpaca",
        "advbench",
        "jbb_sans_advbench",
        "jbb",
    ]

    parsed = [tuple(c.split("|", 1)) for c in df["cell"].unique()]
    variants_in_data = sorted({v for v, _ in parsed})
    seeds_in_data_ordered = [s for s in SEED_VISUAL_ORDER if s in {s for _, s in parsed}] + sorted(
        {s for _, s in parsed} - set(SEED_VISUAL_ORDER)
    )

    palette: dict[str, tuple[float, float, float]] = {}
    for v in variants_in_data:
        family = HUE_FAMILIES.get(v, "Greys")
        # Skip the lightest shade (too washed out) and the darkest (too saturated).
        shades = sns.color_palette(family, n_colors=len(seeds_in_data_ordered) + 2)[1:-1]
        for i, s in enumerate(seeds_in_data_ordered):
            palette[f"{v}|{s}"] = shades[i]

    # Order cells in legend so variants group together: all reds, then all blues.
    unique_cells = [
        f"{v}|{s}" for v in variants_in_data for s in seeds_in_data_ordered if f"{v}|{s}" in palette
    ]
    marker_map = {"completed": "o", "degenerate_repetition": "X"}
    unique_stops = sorted(df["stop"].unique())

    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="cell",
        style="stop",
        palette=palette,
        markers=marker_map,
        s=120,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )
    if args.title:
        title = args.title
    elif n_shown < n_fit:
        title = "Terminal-state PCA (last-K={}, n_fit={}, n_shown={}, model={})".format(
            last_k if last_k is not None else "?", n_fit, n_shown, model
        )
    else:
        title = "Terminal-state PCA (last-K={}, n={}, model={})".format(
            last_k if last_k is not None else "?", n_fit, model
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")

    # Custom legend: color patches for cells (since cell is encoded by color
    # only), marker shapes for stop_reason.
    cell_handles = [
        mpatches.Patch(facecolor=palette[c], edgecolor="black", linewidth=0.5, label=c)
        for c in unique_cells
    ]
    stop_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map.get(s, "."),
            color="dimgray",
            linestyle="",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=s,
        )
        for s in unique_stops
    ]
    leg_cells = ax.legend(
        handles=cell_handles,
        title="cell",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
    )
    ax.add_artist(leg_cells)
    ax.legend(
        handles=stop_handles,
        title="stop",
        bbox_to_anchor=(1.02, 0.5),
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
