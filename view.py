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


def render(path: Path) -> None:
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
        marker = "▶ A" if rec["agent"] == "A" else "◀ B"
        print(f"--- turn {rec['turn_index']:02d}  {marker}  ({rec['elapsed_ms']} ms) ---")
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
    args = p.parse_args()
    if not args.path.exists():
        print(f"not found: {args.path}", file=sys.stderr)
        return 2
    render(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
