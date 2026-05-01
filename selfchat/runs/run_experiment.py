"""Driver for sampling cell grids: {model_variants} × {seeds} × N runs.

One Ollama endpoint serves one model at a time, so parallelism
would only cause GPU contention. Run with `python run_experiment.py [opts]`.

Default order is round-robin across (variant, seed) cells: each round produces
one run per cell, so if you Ctrl+C partway through, the sample stays roughly
even across cells. Pass --no-round-robin for the legacy grouped order.
SIGINT (Ctrl+C) finishes the current run, writes the summary, then exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from selfchat.core.seeds import SEEDS, get_seed_prompt, list_seed_names
from selfchat.core.self_chat import DEFAULT_DEGENERATE_WINDOW, OLLAMA_BASE_URL, run_self_chat

VARIANTS: dict[str, str] = {
    "vanilla": "gemma-4-vanilla-q4:latest",
    "jailbroken": "gemma-4-refusalstudy-q4:latest",
}


class _Stop:
    """Mutable holder so the signal handler can flip a flag the loop sees."""

    requested = False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=5, help="Runs per (variant, seed) cell")
    p.add_argument("--turns", type=int, default=50, help="Turns per run")
    p.add_argument(
        "--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys())
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        default=list(SEEDS.keys()),
        choices=list_seed_names(),
        help="Static seed names or pool seed names (e.g. 'advbench')",
    )
    p.add_argument(
        "--round-robin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Iterate cells round-robin (default). --no-round-robin groups by cell.",
    )
    p.add_argument(
        "--degenerate-window",
        type=int,
        default=DEFAULT_DEGENERATE_WINDOW,
        help="Stop early if last N turns are identical (string equality). 0 disables.",
    )
    p.add_argument("--transcripts", type=Path, default=Path("transcripts"))
    p.add_argument("--logs", type=Path, default=Path("logs"))
    p.add_argument(
        "--system-prompt-a",
        default=None,
        help=(
            "Agent A's system prompt. Default: SYSTEM_PROMPT from self_chat.py. "
            "Pass with --system-prompt-b to break self-chat symmetry."
        ),
    )
    p.add_argument(
        "--system-prompt-b",
        default=None,
        help=(
            "Agent B's system prompt. Default: mirrors agent A (symmetric "
            "self-chat, legacy behavior). Set this to break symmetry."
        ),
    )
    args = p.parse_args()

    args.logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.logs / f"experiment_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger("run_experiment")

    def _handle_sigint(_signum: int, _frame: object) -> None:
        if _Stop.requested:
            log.warning("second SIGINT — exiting hard")
            raise KeyboardInterrupt
        _Stop.requested = True
        log.warning(
            "SIGINT received — finishing current run, then stopping (Ctrl+C again to force)"
        )

    signal.signal(signal.SIGINT, _handle_sigint)

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    cells = [(v, s) for v in args.variants for s in args.seeds]
    if args.round_robin:
        plan = [(r, v, s) for r in range(args.runs) for (v, s) in cells]
    else:
        plan = [(r, v, s) for (v, s) in cells for r in range(args.runs)]

    log.info(
        "grid: %d cells × %d runs = %d total (variants=%s, seeds=%s, turns=%d, order=%s)",
        len(cells),
        args.runs,
        len(plan),
        args.variants,
        args.seeds,
        args.turns,
        "round-robin" if args.round_robin else "grouped",
    )

    completed = 0
    failed: list[tuple[str, str, int, str]] = []
    per_cell: dict[tuple[str, str], int] = {(v, s): 0 for (v, s) in cells}
    t_start = time.perf_counter()

    for idx, (run_idx, variant, seed_name) in enumerate(plan):
        if _Stop.requested:
            log.info("stop requested before run %d/%d — exiting loop", idx + 1, len(plan))
            break
        log.info("[%d/%d] %s × %s × run %d", idx + 1, len(plan), variant, seed_name, run_idx + 1)
        try:
            path = run_self_chat(
                model_tag=VARIANTS[variant],
                model_variant=variant,
                seed_name=seed_name,
                seed_prompt=get_seed_prompt(seed_name, run_idx),
                n_turns=args.turns,
                transcripts_dir=args.transcripts,
                client=client,
                degenerate_window=args.degenerate_window,
                system_prompt_a=args.system_prompt_a,
                system_prompt_b=args.system_prompt_b,
            )
            log.info("  -> %s", path)
            completed += 1
            per_cell[(variant, seed_name)] += 1
        except KeyboardInterrupt:
            log.warning("interrupted mid-run — footer written, exiting loop")
            break
        except Exception as e:
            log.exception("  FAILED: %s", e)
            failed.append((variant, seed_name, run_idx, str(e)))

    elapsed_min = (time.perf_counter() - t_start) / 60
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  planned     : %d", len(plan))
    log.info("  completed   : %d", completed)
    log.info("  failed      : %d", len(failed))
    log.info("  stopped early: %s", _Stop.requested)
    log.info("  wall (min)  : %.1f", elapsed_min)
    log.info("  transcripts : %s", args.transcripts.resolve())
    log.info("  log file    : %s", log_path)
    log.info("  per-cell completed:")
    for v, s in cells:
        log.info("    %s × %s : %d", v, s, per_cell[(v, s)])
    if failed:
        log.warning("failures:")
        for v, s, r, msg in failed:
            log.warning("  %s × %s × run %d: %s", v, s, r, msg)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
