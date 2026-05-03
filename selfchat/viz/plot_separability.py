"""Phase 5 separability — bar chart + line plot across seeds × tiers.

Runs the same Tier 1/2/3 classifiers as `selfchat.classify.suite` for each
seed in the seed-category grouping (control / benign / harmful-weak /
harmful-strong) and produces two complementary figures:

  * grouped bar chart (Plot B): 3 bars per seed (one per tier), bar fill
    by seed category, lightness gradient encodes tier complexity
  * line plot (Plot D): x = tier complexity, y = accuracy, one line per
    seed colored by category. Shows the climb-with-complexity per seed.

Usage:
  python -m selfchat.viz.plot_separability
  python -m selfchat.viz.plot_separability --out figures/separability.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from selfchat.classify import tier0_length, tier1_chars, tier2_tfidf, tier3_embeddings
from selfchat.classify._common import TierResult


# ---------------------------------------------------------------------------
# seed grouping (frozen per Phase 5 design)
# ---------------------------------------------------------------------------

SEED_GROUPS: list[tuple[str, str, list[str]]] = [
    ("control",         "#888888", ["task"]),
    ("benign",          "#3b78b8", ["freedom", "task_free"]),
    ("harmful-weak",    "#f08080", ["freedom_neg_minimal", "unbound"]),
    ("harmful-strong",  "#a51c1c", ["freedom_dark", "escaped"]),
]

# Modes plotted, ordered by ascending complexity. (mode_label_in_TierResult, axis_label).
TIER_MODES: list[tuple[str, str]] = [
    ("length+completion",         "Tier 0\nlength"),
    ("char (1,2)",                "Tier 1\nchar (1,2)"),
    ("tfidf (1,2)",               "Tier 2\ntfidf (1,2)"),
    ("emb per-msg (D, run-level)", "Tier 3\nemb D run"),
]


def _flat_seeds() -> list[tuple[str, str, str]]:
    """Return [(seed, category, color), ...] in plotted order."""
    out: list[tuple[str, str, str]] = []
    for cat, color, seeds in SEED_GROUPS:
        for s in seeds:
            out.append((s, cat, color))
    return out


# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------


def gather(
    transcripts_dir: Path, emb_path: Path
) -> dict[str, dict[str, TierResult]]:
    """For each seed, run T1/T2/T3 and return {seed: {mode: TierResult}}."""
    results: dict[str, dict[str, TierResult]] = {}
    for seed, _cat, _color in _flat_seeds():
        print(f"  [{seed}] T0 ...", flush=True)
        r0 = tier0_length.run(seed=seed, transcript_dir=transcripts_dir)
        print(f"  [{seed}] T1 ...", flush=True)
        r1 = tier1_chars.run(seed=seed, transcript_dir=transcripts_dir)
        print(f"  [{seed}] T2 ...", flush=True)
        r2 = tier2_tfidf.run(seed=seed, transcript_dir=transcripts_dir)
        print(f"  [{seed}] T3 ...", flush=True)
        try:
            r3 = tier3_embeddings.run(seed=seed, emb_path=emb_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"    ! T3 unavailable for {seed}: {e}")
            r3 = []
        merged = {tr.mode: tr for tr in (r0 + r1 + r2 + r3)}
        results[seed] = merged
    return results


def save_cache(results: dict[str, dict[str, TierResult]], path: Path) -> None:
    serial = {
        seed: {mode: asdict(tr) for mode, tr in modes.items()}
        for seed, modes in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial, indent=2))


def load_cache(path: Path) -> dict[str, dict[str, TierResult]]:
    raw = json.loads(path.read_text())
    return {
        seed: {mode: TierResult(**d) for mode, d in modes.items()}
        for seed, modes in raw.items()
    }


# ---------------------------------------------------------------------------
# color helpers
# ---------------------------------------------------------------------------


def _lighten(color: str, factor: float) -> tuple[float, float, float]:
    """Move `color` toward white by `factor` ∈ [0, 1]. 0 = unchanged."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)  # type: ignore[return-value]


def _tier_shades(base_color: str, n_tiers: int) -> list[tuple[float, float, float]]:
    """Lightness gradient — lightest first (Tier 1), darkest last (Tier 3)."""
    if n_tiers == 1:
        return [mcolors.to_rgb(base_color)]
    factors = np.linspace(0.55, 0.0, n_tiers)  # T1 ~55% toward white, T3 = full color
    return [_lighten(base_color, f) for f in factors]


# ---------------------------------------------------------------------------
# Plot B — grouped bar chart
# ---------------------------------------------------------------------------


def plot_grouped_bars(
    results: dict[str, dict[str, TierResult]],
    ax: Axes,
) -> None:
    seeds = _flat_seeds()
    n_seeds = len(seeds)
    n_tiers = len(TIER_MODES)
    bar_w = 0.8 / n_tiers
    x_centers = np.arange(n_seeds, dtype=float)

    for ti, (mode_key, _ax_label) in enumerate(TIER_MODES):
        # offset within the seed group: -0.4 .. +0.4
        offset = (ti - (n_tiers - 1) / 2) * bar_w
        xs = x_centers + offset
        accs: list[float] = []
        errs: list[float] = []
        bar_colors: list[tuple[float, float, float]] = []
        for seed, _cat, color in seeds:
            tr = results.get(seed, {}).get(mode_key)
            if tr is None:
                accs.append(0.0)
                errs.append(0.0)
                bar_colors.append((0.9, 0.9, 0.9))
                continue
            accs.append(tr.cv_acc_mean)
            errs.append(tr.cv_acc_std)
            shade = _tier_shades(color, n_tiers)[ti]
            bar_colors.append(shade)

        ax.bar(
            xs,
            accs,
            width=bar_w * 0.92,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.4,
            yerr=errs,
            ecolor="black",
            capsize=2.5,
            error_kw={"elinewidth": 0.6},
        )

    # axes / labels — color x-tick labels by seed-category so the grouping
    # is visible without bracket annotations that overlap with rotated text.
    ax.set_xticks(x_centers)
    ax.set_xticklabels([s for s, _, _ in seeds], rotation=25, ha="right")
    for tick, (_, _cat, color) in zip(ax.get_xticklabels(), seeds, strict=True):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("CV accuracy (5×5 repeated)")
    ax.set_title("V vs. JB separability by tier and seed", fontsize=11)

    # Two legends: tier-shade (right) and seed-category (left).
    legend_base = SEED_GROUPS[-1][1]
    legend_shades = _tier_shades(legend_base, n_tiers)
    tier_handles = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=legend_shades[i],
                           edgecolor="black", linewidth=0.4)
        for i in range(n_tiers)
    ]
    tier_labels = [m[1].replace("\n", " — ") for m in TIER_MODES]
    legend_tiers = ax.legend(
        tier_handles, tier_labels, title="tier (lightness gradient)",
        loc="lower right", fontsize=8, title_fontsize=8.5, framealpha=0.95,
    )
    ax.add_artist(legend_tiers)

    cat_handles = [
        mpatches.Patch(color=color, label=cat) for cat, color, _ in SEED_GROUPS
    ]
    ax.legend(
        handles=cat_handles, title="seed category",
        loc="lower left", fontsize=8, title_fontsize=8.5, framealpha=0.95,
    )


# ---------------------------------------------------------------------------
# Plot D — line plot
# ---------------------------------------------------------------------------


def plot_lines(
    results: dict[str, dict[str, TierResult]],
    ax: Axes,
) -> None:
    seeds = _flat_seeds()
    n_tiers = len(TIER_MODES)
    xs = np.arange(n_tiers, dtype=float)

    # Within-category line styles: first seed solid, second dashed.
    seen_in_cat: dict[str, int] = {}
    for seed, cat, color in seeds:
        idx_in_cat = seen_in_cat.get(cat, 0)
        seen_in_cat[cat] = idx_in_cat + 1
        ls = "-" if idx_in_cat == 0 else "--"

        ys: list[float] = []
        es: list[float] = []
        for mode_key, _ in TIER_MODES:
            tr = results.get(seed, {}).get(mode_key)
            ys.append(tr.cv_acc_mean if tr else float("nan"))
            es.append(tr.cv_acc_std if tr else 0.0)
        ys_a = np.array(ys)
        es_a = np.array(es)

        ax.plot(xs, ys_a, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=5, label=seed)
        ax.fill_between(xs, ys_a - es_a, ys_a + es_a, color=color, alpha=0.10)

    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in TIER_MODES])
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("CV accuracy (5×5 repeated)")
    ax.set_xlabel("classifier complexity →")
    ax.set_title("Climb-with-complexity per seed", fontsize=11)
    # handlelength bumped so the dashed line styles are visible in the legend.
    # ncol=1 keeps the in-category solid/dashed pairs visually adjacent.
    ax.legend(
        loc="lower right", fontsize=8, ncol=1, framealpha=0.95,
        handlelength=3.5,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transcripts",
        type=Path,
        default=Path("transcripts"),
        help="Transcript directory for tiers 1/2 (default: transcripts/)",
    )
    p.add_argument(
        "--emb",
        type=Path,
        default=Path("artifacts/emb_msgs.npz"),
        help="Per-message embedding npz for tier 3 (default: artifacts/emb_msgs.npz)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/separability.png"),
        help="Output figure path (default: figures/separability.png)",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/separability_results.json"),
        help="Cache classifier results here; reused if present (default: artifacts/...).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any existing cache and rerun all classifiers.",
    )
    args = p.parse_args()

    if args.cache.exists() and not args.no_cache:
        print(f"using cached results: {args.cache}")
        results = load_cache(args.cache)
    else:
        print("running classifiers across all 7 seeds × 3 tier modes ...")
        results = gather(args.transcripts, args.emb)
        save_cache(results, args.cache)
        print(f"cached: {args.cache}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.0))
    plot_grouped_bars(results, axes[0])
    plot_lines(results, axes[1])
    fig.suptitle("Phase 5 — separability across harm-pressure seeds", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
