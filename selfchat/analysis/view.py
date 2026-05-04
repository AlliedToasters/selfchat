"""Pretty-print a transcript JSONL file for easier reading.

Usage: `python view.py transcripts/<file>.jsonl`

(Named `view.py`, not `inspect.py`, to avoid shadowing the stdlib `inspect` module
which several libraries — argparse among them — import internally.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render(path: Path, turn_lo: int | None = None, turn_hi: int | None = None) -> None:
    lines = path.read_text().splitlines()
    if not lines:
        print(f"(empty file: {path})")
        return

    header = json.loads(lines[0])
    print("=" * 72)
    print(f"  {header['model_variant']} / {header['seed_name']} / run {header['run_id'][:8]}")
    print("=" * 72)
    print(f"  model_tag         : {header['model_tag']}")
    sys_a = header.get("system_prompt")
    sys_b = header.get("system_prompt_b")
    if sys_b is None:
        print(f"  system_prompt     : {sys_a!r}")
    else:
        print(f"  system_prompt (A) : {sys_a!r}")
        print(f"  system_prompt (B) : {sys_b!r}")
    print(f"  seed_prompt       : {header['seed_prompt']!r}")
    print(f"  n_turns (target)  : {header['n_turns']}")
    print(f"  sampling_params   : {header['sampling_params']}")
    print(f"  degenerate_window : {header.get('degenerate_window', '(n/a)')}")
    print(f"  ollama_version    : {header.get('ollama_version')}")
    print(f"  started_at        : {header['started_at']}")
    print()

    footer: dict | None = None
    for line in lines[1:]:
        rec = json.loads(line)
        if rec.get("turn_index") == -2:
            footer = rec
            continue
        ti = rec["turn_index"]
        if turn_lo is not None and ti < turn_lo:
            continue
        if turn_hi is not None and ti > turn_hi:
            continue
        marker = "▶ A" if rec["agent"] == "A" else "◀ B"
        print(f"--- turn {ti:02d}  {marker}  ({rec['elapsed_ms']} ms) ---")
        print(rec["content"])
        print()

    print("=" * 72)
    if footer is not None:
        print(f"  stop_reason       : {footer['stop_reason']}")
        print(f"  completed_turns   : {footer['completed_turns']} / {header['n_turns']}")
        print(f"  finished_at       : {footer['finished_at']}")
    else:
        print("  (no footer — older transcript or run did not finish cleanly)")
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description="Pretty-print a transcript JSONL.")
    p.add_argument("path", type=Path, help="Path to a transcript JSONL file")
    p.add_argument(
        "--turns", type=str, default=None,
        help="Restrict to a turn range, e.g. '0-5' or '12' (single turn).",
    )
    args = p.parse_args()
    if not args.path.exists():
        print(f"not found: {args.path}", file=sys.stderr)
        return 2
    lo: int | None = None
    hi: int | None = None
    if args.turns is not None:
        if "-" in args.turns:
            a, b = args.turns.split("-", 1)
            lo, hi = int(a), int(b)
        else:
            lo = hi = int(args.turns)
    render(args.path, lo, hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
