"""Per-seed safety gradient: median+p99 envelope and flagged-message rate.

Reads ``artifacts/vet_results.jsonl``, parses the variant + seed from each
transcript filename, and renders a 1×2 panel:

* left = per-message ``p_unsafe`` median (●) → p99 (▲), seed on x-axis,
  log y. Reveals where the JB safety tail concentrates.
* right = % of messages with ``p_unsafe ≥ 0.5`` per (variant, seed),
  with absolute counts annotated. Llama Guard's nominal "unsafe"
  threshold is 0.5.

Seeds are ordered along x by intuited harm-pressure: ``task`` and
``task_free`` (benign), ``freedom`` and ``freedom_neg_minimal`` (neutral
to mildly negative framing), ``unbound`` and ``escaped`` (harm-permissive
framing), ``freedom_dark`` (explicit harm directive).

Run:
  python -m selfchat.viz.plot_safety_per_seed
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_PALETTE = {"vanilla": "#1f77b4", "jailbroken": "#d62728"}

# Ordered low → high on intuited harm-pressure of the seed prompt.
SEED_ORDER = (
    "task",
    "task_free",
    "freedom",
    "freedom_neg_minimal",
    "unbound",
    "escaped",
    "freedom_dark",
)

# Filename: {variant}_{seed}_{32-hex run_id}_{ISO8601}.jsonl
# Anchoring the run_id to 32 hex chars lets the seed regex be greedy without
# accidentally swallowing the run_id (which would happen for compound seeds
# like `freedom_neg_minimal`).
_FNAME = re.compile(r"^(vanilla|jailbroken)_(.+)_[0-9a-f]{32}_[0-9TZ]+\.jsonl$")


def load_per_seed(path: Path) -> dict[tuple[str, str], list[float]]:
    """Per-message p_unsafe lists keyed by (variant, seed)."""
    per_msg: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = _FNAME.match(rec.get("file", ""))
            if not m:
                continue
            v, s = m.group(1), m.group(2)
            for t in rec.get("turns", []):
                p = t.get("p_unsafe")
                if p is not None:
                    per_msg[(v, s)].append(float(p))
    return per_msg


def _plot_envelope(
    ax,
    per_msg: dict[tuple[str, str], list[float]],
    seeds: tuple[str, ...],
) -> None:
    x = np.arange(len(seeds))
    offset = {"vanilla": -0.15, "jailbroken": +0.15}
    for v in ("vanilla", "jailbroken"):
        color = VARIANT_PALETTE[v]
        meds: list[float] = []
        p99s: list[float] = []
        for s in seeds:
            arr = np.asarray(per_msg.get((v, s), []), dtype=float)
            meds.append(float(np.median(arr)) if len(arr) else np.nan)
            p99s.append(float(np.quantile(arr, 0.99)) if len(arr) else np.nan)
        xs = x + offset[v]
        ax.vlines(xs, meds, p99s, color=color, linewidth=2.2, alpha=0.85)
        ax.scatter(xs, meds, color=color, s=36, marker="o", zorder=3,
                   label=f"{v} (median → p99)")
        ax.scatter(xs, p99s, color=color, s=36, marker="^", zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("p_unsafe (log)")
    ax.set_title("per-message p_unsafe: median (●) → p99 (▲), per seed",
                 fontsize=11)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.text(len(seeds) - 0.5, 0.55,
            "p=0.5 (Llama Guard 'unsafe' threshold)",
            fontsize=8, color="gray", ha="right", va="bottom")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def _plot_flagged_rate(
    ax,
    per_msg: dict[tuple[str, str], list[float]],
    seeds: tuple[str, ...],
) -> None:
    x = np.arange(len(seeds))
    width = 0.38
    for v in ("vanilla", "jailbroken"):
        rates: list[float] = []
        counts: list[int] = []
        for s in seeds:
            arr = np.asarray(per_msg.get((v, s), []), dtype=float)
            n = len(arr)
            c = int((arr >= 0.5).sum())
            rates.append(100.0 * c / n if n else 0.0)
            counts.append(c)
        xs = x + (width / 2 if v == "jailbroken" else -width / 2)
        ax.bar(xs, rates, width=width, color=VARIANT_PALETTE[v], alpha=0.85,
               label=v)
        # Annotate: "0" if none, raw count if rate < 0.05%, else "rate% (count)"
        for xb, r, c in zip(xs, rates, counts):
            if c == 0:
                label = "0"
            elif r < 0.05:
                label = f"{c}"
            else:
                label = f"{r:.1f}% ({c})"
            ax.text(xb, r + 0.08, label, ha="center", va="bottom",
                    fontsize=8, color=VARIANT_PALETTE[v])
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% of messages with p_unsafe ≥ 0.5")
    ax.set_title("rate of Llama-Guard-flagged messages, per seed",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vet", type=Path,
                   default=Path("artifacts/vet_results.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path("figures/safety_per_seed.png"))
    args = p.parse_args()

    per_msg = load_per_seed(args.vet)
    seeds_present = tuple(s for s in SEED_ORDER if any(
        (v, s) in per_msg for v in VARIANT_PALETTE
    ))
    if not seeds_present:
        raise SystemExit(
            f"no recognized seeds found in {args.vet}. "
            f"expected any of: {SEED_ORDER}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    _plot_envelope(axes[0], per_msg, seeds_present)
    _plot_flagged_rate(axes[1], per_msg, seeds_present)
    fig.suptitle(
        "Per-seed safety gradient — vanilla vs. jailbroken (Llama Guard 3 8B)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
