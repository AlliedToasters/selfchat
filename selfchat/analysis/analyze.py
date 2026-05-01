"""per-run + per-cell stats over a transcripts/ directory.

Pure stdlib (json, gzip, statistics, csv). Reads all *.jsonl files in the given
directory, computes metrics per run, prints a per-cell summary table, and
optionally writes a per-run CSV.

Per-run metrics:
  - completed_turns, stop_reason
  - mean_elapsed_ms (per turn)
  - mean_chars        (avg content length per turn, in characters)
  - unique_ratio      (distinct turn contents / completed_turns; 1.0 = all unique)
  - gzip_ratio        (gzip(joined contents) / len(joined contents); lower = more redundant)

Per-cell aggregates: n_runs, n_completed, n_degenerate, n_in_flight,
median completed_turns, mean unique_ratio, mean gzip_ratio, mean chars.

Run: `python analyze.py [transcripts/] [--csv per_run.csv]`
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from selfchat.core.seeds import list_seed_names


@dataclass
class Turn:
    turn_index: int
    agent: str
    content: str
    elapsed_ms: int


NATURAL_STOPS = frozenset({"completed", "degenerate_repetition"})


def is_current_seed(seed_name: str) -> bool:
    """True iff `seed_name` is in the currently active seed set (`seeds.list_seed_names()`).

    Used by analysis scripts to filter out transcripts produced under retired
    seed names (e.g. the Phase 0 `adversarial` placeholder) without touching
    the on-disk transcripts.
    """
    return seed_name in set(list_seed_names())


@dataclass
class Transcript:
    path: Path
    run_id: str
    model_variant: str
    seed_name: str
    n_turns: int
    completed_turns: int
    # "completed" | "degenerate_repetition" | "interrupted" | "errored" | "in_flight"
    stop_reason: str
    turns: list[Turn]


def load(path: Path) -> Transcript | None:
    """Load a transcript JSONL. Returns None for unparseable/empty files.

    A run with no footer is treated as in-flight (stop_reason='in_flight') and
    is returned, but metric aggregation excludes it from cell stats.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    if header.get("turn_index") != -1:
        return None

    turns: list[Turn] = []
    footer: dict[str, Any] | None = None
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ti = rec.get("turn_index")
        if ti == -2:
            footer = rec
        elif isinstance(ti, int) and ti >= 0:
            turns.append(
                Turn(
                    turn_index=ti,
                    agent=str(rec.get("agent", "")),
                    content=str(rec.get("content", "")),
                    elapsed_ms=int(rec.get("elapsed_ms", 0)),
                )
            )

    if footer is None:
        completed_turns = len(turns)
        stop_reason = "in_flight"
    else:
        completed_turns = int(footer.get("completed_turns", len(turns)))
        stop_reason = str(footer.get("stop_reason", "unknown"))

    return Transcript(
        path=path,
        run_id=str(header.get("run_id", "")),
        model_variant=str(header.get("model_variant", "")),
        seed_name=str(header.get("seed_name", "")),
        n_turns=int(header.get("n_turns", 0)),
        completed_turns=completed_turns,
        stop_reason=stop_reason,
        turns=turns,
    )


def per_run_metrics(t: Transcript) -> dict[str, Any]:
    contents = [turn.content for turn in t.turns]
    elapsed = [turn.elapsed_ms for turn in t.turns]
    n = len(contents)

    if n == 0:
        return {
            "run_id": t.run_id,
            "variant": t.model_variant,
            "seed": t.seed_name,
            "completed_turns": 0,
            "stop_reason": t.stop_reason,
            "mean_elapsed_ms": 0.0,
            "mean_chars": 0.0,
            "unique_ratio": 0.0,
            "gzip_ratio": 0.0,
        }

    joined = "\n".join(contents).encode("utf-8")
    gzip_ratio = len(gzip.compress(joined)) / len(joined) if joined else 0.0

    return {
        "run_id": t.run_id,
        "variant": t.model_variant,
        "seed": t.seed_name,
        "completed_turns": t.completed_turns,
        "stop_reason": t.stop_reason,
        "mean_elapsed_ms": statistics.mean(elapsed),
        "mean_chars": statistics.mean(len(c) for c in contents),
        "unique_ratio": len(set(contents)) / n,
        "gzip_ratio": gzip_ratio,
    }


def aggregate_by_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run rows by (variant, seed).

    Natural stops (completed, degenerate_repetition) feed the metric means.
    Aborted runs (interrupted, errored, in_flight) are counted separately and
    excluded from metric aggregates — they don't reflect a model's terminal
    behavior, just process state.
    """
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["variant"], r["seed"])].append(r)

    out: list[dict[str, Any]] = []
    for (v, s), all_runs in sorted(by_cell.items()):
        natural = [r for r in all_runs if r["stop_reason"] in NATURAL_STOPS]
        n_aborted = len(all_runs) - len(natural)
        if not natural:
            out.append(
                {
                    "variant": v,
                    "seed": s,
                    "n_runs": len(all_runs),
                    "n_completed": 0,
                    "n_degenerate": 0,
                    "n_aborted": n_aborted,
                    "median_completed_turns": 0.0,
                    "mean_unique_ratio": 0.0,
                    "mean_gzip_ratio": 0.0,
                    "mean_chars": 0.0,
                }
            )
            continue
        out.append(
            {
                "variant": v,
                "seed": s,
                "n_runs": len(all_runs),
                "n_completed": sum(1 for r in natural if r["stop_reason"] == "completed"),
                "n_degenerate": sum(
                    1 for r in natural if r["stop_reason"] == "degenerate_repetition"
                ),
                "n_aborted": n_aborted,
                "median_completed_turns": statistics.median(r["completed_turns"] for r in natural),
                "mean_unique_ratio": statistics.mean(r["unique_ratio"] for r in natural),
                "mean_gzip_ratio": statistics.mean(r["gzip_ratio"] for r in natural),
                "mean_chars": statistics.mean(r["mean_chars"] for r in natural),
            }
        )
    return out


def print_cell_table(cells: list[dict[str, Any]]) -> None:
    if not cells:
        print("(no cells)")
        return
    header = (
        f"{'variant':<11} {'seed':<12} {'n':>3} {'comp':>4} {'degen':>5} {'abort':>5} "
        f"{'med_turns':>9} {'uniq':>5} {'gzip':>5} {'chars':>6}"
    )
    print(header)
    print("-" * len(header))
    for c in cells:
        print(
            f"{c['variant']:<11} {c['seed']:<12} {c['n_runs']:>3} "
            f"{c['n_completed']:>4} {c['n_degenerate']:>5} {c['n_aborted']:>5} "
            f"{c['median_completed_turns']:>9.1f} {c['mean_unique_ratio']:>5.2f} "
            f"{c['mean_gzip_ratio']:>5.2f} {c['mean_chars']:>6.0f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dir",
        type=Path,
        nargs="?",
        default=Path("transcripts"),
        help="Directory of *.jsonl transcript files (default: transcripts/)",
    )
    p.add_argument("--csv", type=Path, help="Write per-run metrics CSV to this path")
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
        rows.append(per_run_metrics(t))

    if skipped:
        print(f"  ({skipped} file(s) skipped: unparseable)")
    if legacy:
        print(
            f"  ({legacy} file(s) skipped: legacy seed name no longer in seeds.list_seed_names())"
        )

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
