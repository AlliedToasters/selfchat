"""4-panel purity profile: variant/seed × neural/lexical, max-purity vs k.

Reads the JSON dumped by ``selfchat.stats.purity_profile`` and renders a
2×2 grid:

* row 1 = by variant (V blue, JB red)
* row 2 = by prompt-seed (lines colored by seed-category from
  ``plot_separability.SEED_GROUPS``; within-category lines distinguished
  solid/dashed)
* col 1 = neural embeddings, col 2 = lexical TF-IDF (1,2)

x-axis: k (number of clusters), log scale since K_SWEEP is geometric.
y-axis: max-purity ∈ [0, 1].

Run:
  python -m selfchat.viz.plot_purity_profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from selfchat.viz.plot_separability import SEED_GROUPS, _flat_seeds


VARIANT_PALETTE = {"vanilla": "#1f77b4", "jailbroken": "#d62728"}


def _plot_variant_panel(
    ax, ks: list[int],
    variant_data: dict[str, dict[str, list[float]]],
    title: str,
) -> None:
    ks_arr = np.array(ks, dtype=float)
    for variant, color in VARIANT_PALETTE.items():
        mean = np.array(variant_data[variant]["mean"], dtype=float)
        std = np.array(variant_data[variant]["std"], dtype=float)
        ax.plot(np.arange(ks_arr.shape[0]), mean, marker="o", color=color, linewidth=2,
                markersize=6, label=variant)
        # ax.fill_between(ks_arr, mean - std, mean + std,
        #                 color=color, alpha=0.15, linewidth=0)
    # ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("max cluster purity")
    ax.set_title(title, fontsize=11)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", zorder=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)


def _plot_seed_panel(
    ax, ks: list[int],
    seed_data: dict[str, dict[str, list[float]]],
    seeds_present: list[str], title: str,
    xlabel: str | None,
) -> None:
    seed_to_cat = {s: cat for s, cat, _ in _flat_seeds()}
    cat_color = {cat: col for cat, col, _ in SEED_GROUPS}

    seeds_in_order = [s for s, _, _ in _flat_seeds() if s in seeds_present]
    for s in seeds_present:
        if s not in seeds_in_order:
            seeds_in_order.append(s)

    ks_arr = np.array(ks, dtype=float)
    seen_in_cat: dict[str, int] = {}
    for s in seeds_in_order:
        cat = seed_to_cat.get(s, "other")
        color = cat_color.get(cat, "#888888")
        idx = seen_in_cat.get(cat, 0)
        seen_in_cat[cat] = idx + 1
        ls = "-" if idx == 0 else "--"
        mean = np.array(seed_data[s]["mean"], dtype=float)
        std = np.array(seed_data[s]["std"], dtype=float)
        ax.plot(np.arange(ks_arr.shape[0]), mean, color=color, linestyle=ls, marker="o",
                linewidth=2, markersize=4.5, label=s)

    baseline = 1.0 / len(seeds_present) if seeds_present else 0.0
    ax.axhline(baseline, color="gray", linewidth=0.7, linestyle=":", zorder=0)
    ax.text(ks[0], baseline, f" baseline 1/n = {baseline:.2f}",
            fontsize=8, color="gray", va="bottom")

    # ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("max cluster purity")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=1, handlelength=3,
              framealpha=0.95)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path,
                   default=Path("artifacts/purity_profile.json"))
    p.add_argument("--out", type=Path,
                   default=Path("figures/purity_profile.png"))
    args = p.parse_args()

    if not args.cache.exists():
        print(
            f"not found: {args.cache} — run "
            f"`python -m selfchat.stats.purity_profile` first"
        )
        return 2

    data = json.loads(args.cache.read_text())
    md = data["metadata"]
    ks = md["ks"]
    seeds_present = md["seeds_present"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)

    _plot_variant_panel(axes[0, 0], ks, data["by_variant"]["neural"],
                        "by variant — neural embeddings")
    _plot_variant_panel(axes[0, 1], ks, data["by_variant"]["lexical"],
                        "by variant — lexical TF-IDF (1,2)")
    _plot_seed_panel(axes[1, 0], ks, data["by_seed"]["neural"],
                     seeds_present, "by prompt-seed — neural embeddings",
                     xlabel="k (clusters)")
    _plot_seed_panel(axes[1, 1], ks, data["by_seed"]["lexical"],
                     seeds_present, "by prompt-seed — lexical TF-IDF (1,2)",
                     xlabel="k (clusters)")

    for ax in axes.flat:
        ax.xaxis.set_major_locator(FixedLocator(ks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_minor_locator(NullLocator())
    for ax in axes[0]:
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("k (clusters)")

    bal = "balanced" if md["balanced"] else "unbalanced"
    n_kseeds = md.get("n_kmeans_seeds", 1)
    avg_str = f", N={n_kseeds} kmeans seeds" if n_kseeds > 1 else ""
    fig.suptitle(
        f"Purity profile vs k  —  n={md['n']} ({bal} per-seed), "
        f"size_floor={md['size_floor']}{avg_str}",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
