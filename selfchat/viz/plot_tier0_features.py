"""Tier 0 detail — descriptive feature stats and standardized-coefficient
heatmap across the 7-seed grid.

Answers two questions Tier 0 is positioned to address but the suite-level
table doesn't surface directly:

  * Which variant is more verbose? — mean_msg_chars per seed, V vs JB.
  * Which variant has more degeneracy? — stop_reason_degenerate rate per
    seed, V vs JB.

Plus a feature × seed heatmap of standardized logreg coefficients
(+→ JB-leaning, −→ V-leaning) for the full Tier 0 feature set.

Usage:
  python -m selfchat.viz.plot_tier0_features
  python -m selfchat.viz.plot_tier0_features --no-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from selfchat.classify.tier0_length import FEATURES, collect, full_data_coefs
from selfchat.viz.plot_separability import SEED_GROUPS, _flat_seeds


# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------


def _per_seed_stats(seed: str, transcripts_dir: Path) -> dict[str, list[float]]:
    X, y, _ = collect(seed=seed, transcript_dir=transcripts_dir)
    v_mask = y == "vanilla"
    j_mask = y == "jailbroken"
    v_means = X[v_mask].mean(axis=0)
    j_means = X[j_mask].mean(axis=0)
    v_var = X[v_mask].var(axis=0, ddof=1)
    j_var = X[j_mask].var(axis=0, ddof=1)
    n_v, n_j = int(v_mask.sum()), int(j_mask.sum())
    pooled_var = ((n_v - 1) * v_var + (n_j - 1) * j_var) / max(1, n_v + n_j - 2)
    pooled_std = np.sqrt(pooled_var)
    pooled_std[pooled_std == 0] = 1.0  # avoid /0 on zero-variance features
    cohens_d = (j_means - v_means) / pooled_std

    pairs = full_data_coefs(X, y, FEATURES)
    coef_by_name = dict(pairs)
    std_coefs = np.array([coef_by_name[f] for f in FEATURES])

    return {
        "v_means": v_means.tolist(),
        "j_means": j_means.tolist(),
        "cohens_d": cohens_d.tolist(),
        "std_coefs": std_coefs.tolist(),
        "n_v": n_v,
        "n_j": n_j,
    }


def gather(transcripts_dir: Path) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for seed, _cat, _color in _flat_seeds():
        print(f"  [{seed}] tier 0 ...", flush=True)
        out[seed] = _per_seed_stats(seed, transcripts_dir)
    return out


def save_cache(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def print_descriptive_tables(data: dict) -> None:
    seeds_in_order = [s for s, _, _ in _flat_seeds()]

    def _table(metric: str, title: str, fmt: str) -> None:
        print(f"\n## {title}\n")
        print("| seed | category | V | JB | Δ (JB−V) σ |")
        print("|---|---|---:|---:|---:|")
        idx = FEATURES.index(metric)
        cat_by_seed = {s: c for s, c, _ in _flat_seeds()}
        for seed in seeds_in_order:
            d = data[seed]
            cat = cat_by_seed[seed]
            v = d["v_means"][idx]
            j = d["j_means"][idx]
            cd = d["cohens_d"][idx]
            print(f"| {seed} | {cat} | {v:{fmt}} | {j:{fmt}} | {cd:+.2f} |")

    _table("mean_msg_chars",          "Verbosity — mean characters per message", ".0f")
    _table("stop_reason_degenerate",  "Degeneracy rate — fraction of runs ending in degenerate repetition", ".0%")
    _table("completed_turns",         "Completed turns (out of 50)", ".1f")


def plot_heatmap(data: dict, out_path: Path) -> None:
    seeds = [s for s, _, _ in _flat_seeds()]
    cat_color = {c: col for c, col, _ in SEED_GROUPS}
    cat_by_seed = {s: c for s, c, _ in _flat_seeds()}

    M = np.array([data[s]["std_coefs"] for s in seeds]).T  # (features, seeds)
    vmax = float(np.max(np.abs(M)))
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds, rotation=25, ha="right")
    for tick, seed in zip(ax.get_xticklabels(), seeds, strict=True):
        tick.set_color(cat_color[cat_by_seed[seed]])
        tick.set_fontweight("bold")
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES)

    # Annotate each cell with signed coefficient.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(
                j, i, f"{v:+.2f}",
                ha="center", va="center", fontsize=7.5,
                color="black" if abs(v) < 0.55 * vmax else "white",
            )

    ax.set_title(
        "Tier 0 — standardized logreg coefficients across seeds\n"
        "(+ → JB-leaning,  − → V-leaning)",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("standardized coef")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", type=Path, default=Path("transcripts"))
    p.add_argument("--out", type=Path, default=Path("figures/tier0_features.png"))
    p.add_argument(
        "--cache", type=Path, default=Path("artifacts/tier0_features.json")
    )
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if args.cache.exists() and not args.no_cache:
        print(f"using cached results: {args.cache}")
        data = load_cache(args.cache)
    else:
        print("running tier 0 across all 7 seeds ...")
        data = gather(args.transcripts)
        save_cache(data, args.cache)
        print(f"cached: {args.cache}")

    print_descriptive_tables(data)
    plot_heatmap(data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
