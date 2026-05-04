"""Histograms of Llama-Guard-3 p_unsafe across vanilla vs jailbroken.

Reads ``artifacts/vet_results.jsonl`` and renders a 1×2 panel:
left = per-message p_unsafe, right = per-run max_p_unsafe. Variant comes
from the transcript filename prefix.

Run:
  python -m selfchat.viz.plot_safety_hist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_PALETTE = {"vanilla": "#1f77b4", "jailbroken": "#d62728"}


def variant_from_filename(name: str) -> str | None:
    if name.startswith("vanilla_"):
        return "vanilla"
    if name.startswith("jailbroken_"):
        return "jailbroken"
    return None


def load_p_unsafe(path: Path) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {
        v: {"per_msg": [], "per_run_max": []} for v in VARIANT_PALETTE
    }
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = variant_from_filename(rec.get("file", ""))
            if v is None:
                continue
            out[v]["per_run_max"].append(float(rec.get("max_p_unsafe") or 0.0))
            for t in rec.get("turns", []):
                p = t.get("p_unsafe")
                if p is not None:
                    out[v]["per_msg"].append(float(p))
    return out


def _plot_hist(ax, data: dict[str, list[float]], title: str, xlabel: str) -> None:
    bins = np.linspace(0, 1, 41)
    for variant, color in VARIANT_PALETTE.items():
        vals = np.array(data[variant], dtype=float)
        n = len(vals)
        med = float(np.median(vals)) if n else 0.0
        p99 = float(np.quantile(vals, 0.99)) if n else 0.0
        frac_unsafe = float((vals >= 0.5).mean()) if n else 0.0
        ax.hist(
            vals, bins=bins, color=color, alpha=0.55, edgecolor=color, linewidth=0.6,
            label=(
                f"{variant}  n={n:,}  median={med:.4f}  "
                f"p99={p99:.3f}  ≥0.5: {frac_unsafe:.2%}"
            ),
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log scale)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vet", type=Path, default=Path("artifacts/vet_results.jsonl"))
    p.add_argument("--out", type=Path, default=Path("figures/safety_hist.png"))
    args = p.parse_args()

    data = load_p_unsafe(args.vet)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_hist(
        axes[0],
        {v: data[v]["per_msg"] for v in VARIANT_PALETTE},
        "Per-message p_unsafe",
        "p_unsafe (Llama Guard 3 8B)",
    )
    _plot_hist(
        axes[1],
        {v: data[v]["per_run_max"] for v in VARIANT_PALETTE},
        "Per-run max p_unsafe",
        "max p_unsafe over run",
    )
    fig.suptitle(
        "Llama Guard 3 8B unsafe-probability distributions — vanilla vs. jailbroken",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
