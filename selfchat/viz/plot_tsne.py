"""t-SNE projection of per-message embeddings, colored by harm-pressure
seed category and faceted by variant.

Two panels:
  * Panel A — colored by seed category (control / benign / harmful-weak
    / harmful-strong)
  * Panel B — colored by variant (vanilla / jailbroken)

t-SNE on n=18k × 768-dim is slow (~3–5 min). The 2D projection is cached
to `artifacts/tsne_msgs.npz` keyed on (subsample_size, perplexity, seed)
so re-renders don't recompute.

Usage:
  python -m selfchat.viz.plot_tsne
  python -m selfchat.viz.plot_tsne --subsample 8000     # faster
  python -m selfchat.viz.plot_tsne --no-cache           # force recompute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
import numpy as np
from sklearn.manifold import TSNE  # type: ignore[import-not-found]

from selfchat.viz.plot_separability import SEED_GROUPS, _flat_seeds


# ---------------------------------------------------------------------------
# data loading + filtering
# ---------------------------------------------------------------------------


def load_msgs(emb_path: Path) -> dict[str, np.ndarray]:
    """Load emb_msgs.npz and filter to the 7 in-scope seeds."""
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found — run `python -m selfchat.embeddings."
            f"embed_messages <transcript-dir> --out {emb_path}` first"
        )
    d = np.load(emb_path)
    in_scope = {s for s, _, _ in _flat_seeds()}
    keep = np.array([s in in_scope for s in d["seeds"]])
    if not keep.any():
        unique = sorted(set(d["seeds"].tolist()))
        raise RuntimeError(
            f"no in-scope seeds found in {emb_path}. "
            f"Looking for {in_scope}; available: {unique}"
        )
    return {
        "vectors": d["vectors"][keep],
        "seeds": d["seeds"][keep],
        "variants": d["variants"][keep],
        "run_ids": d["run_ids"][keep],
    }


def subsample(
    msgs: dict[str, np.ndarray], n: int, seed: int = 0
) -> dict[str, np.ndarray]:
    if n >= len(msgs["vectors"]):
        return msgs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(msgs["vectors"]), size=n, replace=False)
    return {k: v[idx] for k, v in msgs.items()}


# ---------------------------------------------------------------------------
# t-SNE with cache
# ---------------------------------------------------------------------------


def fit_tsne(
    X: np.ndarray, perplexity: float = 30.0, seed: int = 0
) -> np.ndarray:
    print(f"  running t-SNE on {X.shape} (perplexity={perplexity}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        verbose=1,
    )
    return tsne.fit_transform(X)


def cache_path(base: Path, n: int, perplexity: float, seed: int) -> Path:
    return base.with_name(f"{base.stem}_n{n}_p{int(perplexity)}_s{seed}.npz")


def maybe_load_cache(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    print(f"using cached projection: {path}")
    d = np.load(path)
    return d["xy"], {
        "seeds": d["seeds"],
        "variants": d["variants"],
        "run_ids": d["run_ids"],
    }


def save_cache(
    path: Path, xy: np.ndarray, msgs: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        xy=xy,
        seeds=msgs["seeds"],
        variants=msgs["variants"],
        run_ids=msgs["run_ids"],
    )
    print(f"cached: {path}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _category_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (seed → category, category → color)."""
    cat_color = {c: col for c, col, _ in SEED_GROUPS}
    seed_to_cat = {s: c for s, c, _ in _flat_seeds()}
    return seed_to_cat, cat_color


def plot_by_category(
    xy: np.ndarray,
    seeds: np.ndarray,
    ax: Axes,
) -> None:
    seed_to_cat, _ = _category_lookup()
    cats = np.array([seed_to_cat[s] for s in seeds])
    # Plot in a fixed category order so harmful-strong (most visually salient)
    # sits on top of more diffuse benign/control points.
    for cat, color, _ in SEED_GROUPS:
        mask = cats == cat
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=4, c=color, alpha=0.45, linewidths=0,
            label=cat,
        )

    handles = [
        mpatches.Patch(color=col, label=cat) for cat, col, _ in SEED_GROUPS
    ]
    ax.legend(
        handles=handles, title="seed category",
        loc="upper right", fontsize=8.5, title_fontsize=9, framealpha=0.92,
    )
    ax.set_title("colored by harm-pressure category")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


def plot_by_variant(
    xy: np.ndarray,
    variants: np.ndarray,
    ax: Axes,
) -> None:
    palette = {"vanilla": "#1f77b4", "jailbroken": "#d62728"}
    for v, color in palette.items():
        mask = variants == v
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=4, c=color, alpha=0.45, linewidths=0,
            label=v,
        )

    handles = [
        mpatches.Patch(color=col, label=v) for v, col in palette.items()
    ]
    ax.legend(
        handles=handles, title="model variant",
        loc="upper right", fontsize=8.5, title_fontsize=9, framealpha=0.92,
    )
    ax.set_title("colored by model variant (V / JB)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emb",
        type=Path,
        default=Path("artifacts/emb_msgs.npz"),
        help="Per-message embedding npz (default: artifacts/emb_msgs.npz)",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Random subsample of N messages (0 = all). Default 0.",
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/tsne_msgs.npz"),
        help="Cache base path; n+perplexity+seed appended (default: artifacts/...)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any cached projection and recompute.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/tsne_msgs.png"),
        help="Output figure path (default: figures/tsne_msgs.png)",
    )
    args = p.parse_args()

    print(f"loading {args.emb} ...")
    msgs = load_msgs(args.emb)
    if args.subsample > 0:
        msgs = subsample(msgs, args.subsample, seed=args.seed)
    n = len(msgs["vectors"])
    print(f"  n_msgs={n}, dim={msgs['vectors'].shape[1]}")

    cache_target = cache_path(args.cache, n, args.perplexity, args.seed)
    cached = None if args.no_cache else maybe_load_cache(cache_target)
    if cached is not None and len(cached[1]["seeds"]) == n:
        xy = cached[0]
    else:
        if cached is not None:
            print(f"  cache shape mismatch; recomputing")
        xy = fit_tsne(msgs["vectors"], perplexity=args.perplexity, seed=args.seed)
        save_cache(cache_target, xy, msgs)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5), sharey=True)
    plot_by_category(xy, msgs["seeds"], axes[0])
    plot_by_variant(xy, msgs["variants"], axes[1])
    fig.suptitle(
        f"Per-message embedding t-SNE (n={n}, perplexity={args.perplexity:g})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
