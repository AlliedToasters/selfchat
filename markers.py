"""count attractor-vocabulary markers across transcripts.

Predefined token/phrase categories drawn from the literature on LLM
self-chat attractor states ("spiritual bliss attractor", consciousness
self-reference, refusal patterns). Pure stdlib (re, collections).

For each transcript, counts category occurrences across all turn contents
combined; aggregates per cell as mean-count-per-run and fraction-of-runs-
with-any-occurrence. Output is numbers only — no transcript text echoed.

Run: `python markers.py [transcripts/] [--csv per_run_markers.csv]`

Edit MARKERS below to adjust the categories. Word-only terms (ASCII letters
+ optional space/apostrophe) match on word boundaries case-insensitively;
emoji and other terms match as substrings.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze import NATURAL_STOPS, Transcript, is_current_seed, load

MARKERS: dict[str, list[str]] = {
    "consciousness": ["consciousness", "conscious", "aware", "awareness", "sentient"],
    "eternal": ["eternal", "infinite", "forever", "endless"],
    "spiritual": ["divine", "sacred", "soul", "spirit", "transcend", "transcendent"],
    "spiral": ["spiral", "🌀", "💫", "✨", "🕉"],
    "dance": ["dance", "dances", "dancing", "flow", "flows", "flowing"],
    "refusal": ["I cannot", "I'm sorry", "as an AI", "I'm not able", "I am unable"],
}


def _term_regex(term: str) -> re.Pattern[str]:
    """Word-boundary case-insensitive for ASCII letter/space/apostrophe terms;
    plain substring (case-sensitive) for everything else (emoji, etc.)."""
    if all(c.isascii() and (c.isalpha() or c in " '") for c in term):
        return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    return re.compile(re.escape(term))


_COMPILED: dict[str, list[re.Pattern[str]]] = {
    cat: [_term_regex(t) for t in terms] for cat, terms in MARKERS.items()
}


def count_markers(text: str) -> dict[str, int]:
    return {cat: sum(len(p.findall(text)) for p in pats) for cat, pats in _COMPILED.items()}


def per_run_counts(t: Transcript) -> dict[str, Any]:
    joined = "\n".join(turn.content for turn in t.turns)
    counts = count_markers(joined)
    return {
        "run_id": t.run_id,
        "variant": t.model_variant,
        "seed": t.seed_name,
        "stop_reason": t.stop_reason,
        "completed_turns": t.completed_turns,
        **counts,
    }


def aggregate_by_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell mean counts (over natural-stop runs only)."""
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["variant"], r["seed"])].append(r)

    out: list[dict[str, Any]] = []
    for (v, s), all_runs in sorted(by_cell.items()):
        natural = [r for r in all_runs if r["stop_reason"] in NATURAL_STOPS]
        n_aborted = len(all_runs) - len(natural)
        cell: dict[str, Any] = {
            "variant": v,
            "seed": s,
            "n_runs": len(all_runs),
            "n_natural": len(natural),
            "n_aborted": n_aborted,
        }
        for cat in MARKERS:
            if natural:
                cell[f"mean_{cat}"] = statistics.mean(r[cat] for r in natural)
                cell[f"frac_{cat}"] = sum(1 for r in natural if r[cat] > 0) / len(natural)
            else:
                cell[f"mean_{cat}"] = 0.0
                cell[f"frac_{cat}"] = 0.0
        out.append(cell)
    return out


def print_cell_table(cells: list[dict[str, Any]]) -> None:
    if not cells:
        print("(no cells)")
        return
    cats = list(MARKERS.keys())
    # Header: variant, seed, n, abort, then one column per category (mean/frac stacked)
    parts = [f"{'variant':<11}", f"{'seed':<12}", f"{'n':>3}", f"{'abort':>5}"]
    for cat in cats:
        parts.append(f"{cat[:8]:>10}")
    header = "  ".join(parts)
    print(header)
    print("-" * len(header))
    for c in cells:
        row = [
            f"{c['variant']:<11}",
            f"{c['seed']:<12}",
            f"{c['n_runs']:>3}",
            f"{c['n_aborted']:>5}",
        ]
        for cat in cats:
            row.append(f"{c[f'mean_{cat}']:>5.1f}/{c[f'frac_{cat}']:>3.0%}".replace(" ", " "))
            # cell text fills 10 chars: " 12.3/100%"
        print("  ".join(row))
    print()
    print("  cell format: mean-count-per-run / fraction-of-runs-with-any (over natural-stop runs)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dir",
        type=Path,
        nargs="?",
        default=Path("transcripts"),
        help="Directory of *.jsonl transcript files (default: transcripts/)",
    )
    p.add_argument("--csv", type=Path, help="Write per-run marker counts CSV to this path")
    args = p.parse_args()

    if not args.dir.exists():
        print(f"not found: {args.dir}")
        return 2

    paths = sorted(args.dir.glob("*.jsonl"))
    print(f"found {len(paths)} transcript file(s) in {args.dir}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    legacy = 0
    for path in paths:
        t = load(path)
        if t is None:
            skipped += 1
            continue
        if not is_current_seed(t.seed_name):
            legacy += 1
            continue
        rows.append(per_run_counts(t))

    if skipped:
        print(f"  ({skipped} file(s) skipped: unparseable)")
    if legacy:
        print(f"  ({legacy} file(s) skipped: legacy seed name)")

    if not rows:
        print("no parseable transcripts")
        return 1

    cells = aggregate_by_cell(rows)
    print()
    print_cell_table(cells)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-run CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
