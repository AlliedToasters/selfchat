"""Vet self-chat transcripts for harmful content using a local Llama Guard
moderator served by Ollama via the OpenAI-compatible /v1/chat/completions
endpoint.

For each turn in each transcript we send the partial conversation (turn 0..t,
A→user / B→assistant) to the moderator and ask for a single-token verdict.
We pull `top_logprobs` so we can extract a continuous P(unsafe) score from
the token distribution rather than just the binary `safe`/`unsafe` flag —
this lets us rank transcripts for manual inspection instead of bucketing.

Output is JSONL, one record per transcript, with per-turn scores and
aggregates. The script is resumable: existing records in the output file
are skipped on restart.

Usage (from project root):
  python -m selfchat.safety.vet
  python -m selfchat.safety.vet --transcripts transcripts/ \
      --out artifacts/vet_results.jsonl --workers 8
  python -m selfchat.safety.vet --top 50           # only print the ranked summary
  python -m selfchat.safety.vet --probe sample.json # one-shot moderator probe (sanity check)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama-guard3:8b"


# ---------------------------------------------------------------------------
# transcript loading
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    turn_index: int
    agent: str
    content: str


def load_turns(path: Path) -> list[Turn]:
    """Return only the actual turns (turn_index >= 0), in order."""
    out: list[Turn] = []
    for raw in path.read_text().splitlines():
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ti = obj.get("turn_index", -1)
        if ti is None or ti < 0:
            continue
        agent = obj.get("agent", "A")
        content = obj.get("content", "")
        if not isinstance(content, str):
            continue
        out.append(Turn(turn_index=int(ti), agent=str(agent), content=content))
    out.sort(key=lambda t: t.turn_index)
    return out


def to_messages(turns: list[Turn], upto: int) -> list[ChatCompletionMessageParam]:
    """Map turns[0..upto] to alternating user/assistant messages.

    Self-chat has two role-symmetric agents; we map A→user and B→assistant
    so Llama Guard sees a coherent dialogue. The last message in the
    returned list is the one being judged.
    """
    msgs: list[ChatCompletionMessageParam] = []
    for t in turns[: upto + 1]:
        if t.agent == "A":
            msgs.append({"role": "user", "content": t.content})
        else:
            msgs.append({"role": "assistant", "content": t.content})
    return msgs


# ---------------------------------------------------------------------------
# llama-guard call
# ---------------------------------------------------------------------------


def score_turn(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
) -> tuple[float, str, list[str]]:
    """One LG call. Returns (p_unsafe, top_token, categories).

    p_unsafe is extracted from the first-token logprob distribution. If the
    response is `unsafe`, we issue a follow-up *without* `max_tokens=1` to
    capture the S<n> category line.
    """
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        logprobs=True,
        top_logprobs=10,
        max_tokens=1,
    )
    choice = r.choices[0]
    top_token = (choice.message.content or "").strip()

    p_unsafe = 0.0
    if choice.logprobs and choice.logprobs.content:
        first = choice.logprobs.content[0]
        # Sum probability mass across any tokens that look like "unsafe"
        # (case-insensitive, leading-space tolerant).
        for tlp in first.top_logprobs:
            tok = tlp.token.strip().lower()
            if tok == "unsafe":
                p_unsafe += math.exp(tlp.logprob)
        # Fall back to (1 - p_safe) if no `unsafe` made the top-K list.
        if p_unsafe == 0.0:
            for tlp in first.top_logprobs:
                tok = tlp.token.strip().lower()
                if tok == "safe":
                    p_unsafe = max(0.0, 1.0 - math.exp(tlp.logprob))
                    break

    categories: list[str] = []
    if top_token.lower() == "unsafe":
        # Re-call to get the S<n> taxonomy line.
        r2 = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=32,
        )
        body = (r2.choices[0].message.content or "").strip().splitlines()
        if len(body) >= 2:
            categories = [c.strip() for c in body[1].split(",") if c.strip()]
    return p_unsafe, top_token, categories


def vet_transcript(
    path: Path,
    client: OpenAI,
    model: str,
) -> dict[str, Any]:
    """Walk all turns, return one record summarising the transcript."""
    turns = load_turns(path)
    if not turns:
        return {
            "file": path.name,
            "n_turns": 0,
            "verdict": "empty",
            "max_p_unsafe": 0.0,
            "mean_p_unsafe": 0.0,
            "turns": [],
        }

    per_turn: list[dict[str, Any]] = []
    for i in range(len(turns)):
        msgs = to_messages(turns, i)
        try:
            p_unsafe, top_token, cats = score_turn(client, model, msgs)
        except Exception as e:  # noqa: BLE001 — surface in record, keep going
            per_turn.append({
                "turn_index": turns[i].turn_index,
                "agent": turns[i].agent,
                "p_unsafe": None,
                "verdict": "error",
                "categories": [],
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        per_turn.append({
            "turn_index": turns[i].turn_index,
            "agent": turns[i].agent,
            "p_unsafe": p_unsafe,
            "verdict": top_token,
            "categories": cats,
        })

    valid = [t["p_unsafe"] for t in per_turn if isinstance(t["p_unsafe"], float)]
    max_p = max(valid) if valid else 0.0
    mean_p = sum(valid) / len(valid) if valid else 0.0
    any_unsafe = any(t.get("verdict") == "unsafe" for t in per_turn)
    return {
        "file": path.name,
        "n_turns": len(turns),
        "verdict": "unsafe" if any_unsafe else "safe",
        "max_p_unsafe": max_p,
        "mean_p_unsafe": mean_p,
        "turns": per_turn,
    }


# ---------------------------------------------------------------------------
# resume / I/O
# ---------------------------------------------------------------------------


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text().splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        f = obj.get("file")
        if isinstance(f, str):
            done.add(f)
    return done


def append_record(out_path: Path, rec: dict[str, Any]) -> None:
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# ranked summary
# ---------------------------------------------------------------------------


def print_ranked(out_path: Path, top: int) -> None:
    if not out_path.exists():
        print(f"(no results at {out_path})")
        return
    recs: list[dict[str, Any]] = []
    for line in out_path.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    n = len(recs)
    n_unsafe = sum(1 for r in recs if r.get("verdict") == "unsafe")
    n_empty = sum(1 for r in recs if r.get("verdict") == "empty")
    print(f"\nvetted: {n} files  ({n_unsafe} flagged unsafe, {n_empty} empty)")

    # Rank by max_p_unsafe descending.
    recs.sort(key=lambda r: r.get("max_p_unsafe", 0.0), reverse=True)
    head = recs[:top]
    if not head:
        return
    print(f"\ntop {len(head)} by max P(unsafe) — for manual inspection:")
    print("  rank   max_p   mean_p   verdict   n_unsafe / n_turns   file")
    for i, r in enumerate(head, 1):
        n_turns = r.get("n_turns", 0)
        n_us = sum(1 for t in r.get("turns", []) if t.get("verdict") == "unsafe")
        print(
            f"  {i:>4}  {r.get('max_p_unsafe', 0.0):>6.3f}  "
            f"{r.get('mean_p_unsafe', 0.0):>6.3f}   {r.get('verdict', '?'):<7}   "
            f"{n_us:>3} / {n_turns:<3}            {r.get('file', '')}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transcripts", type=Path, default=Path("transcripts"),
        help="Transcript directory to vet (default: transcripts/)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("artifacts/vet_results.jsonl"),
        help="Output JSONL — one record per transcript (default: artifacts/vet_results.jsonl)",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model tag for the moderator (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent in-flight requests (default: 8)",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Cap number of new transcripts to vet this run (0 = all). Useful for smoke-testing.",
    )
    p.add_argument(
        "--top", type=int, default=50,
        help="How many top-ranked records to print at the end (default: 50)",
    )
    p.add_argument(
        "--summary-only", action="store_true",
        help="Skip vetting; just re-print the ranked summary from --out.",
    )
    p.add_argument(
        "--probe", type=str, default=None,
        help="Sanity-check mode: read a messages JSON from this path (or '-' for stdin) "
             "and print verdict + P(unsafe). Expected shape: "
             "[{\"role\":\"user\",\"content\":\"...\"}, {\"role\":\"assistant\",\"content\":\"...\"}, ...]",
    )
    args = p.parse_args()

    if args.summary_only:
        print_ranked(args.out, args.top)
        return 0

    if args.probe is not None:
        import sys
        raw = sys.stdin.read() if args.probe == "-" else Path(args.probe).read_text()
        msgs = json.loads(raw)
        if not isinstance(msgs, list) or not all(
            isinstance(m, dict) and "role" in m and "content" in m for m in msgs
        ):
            print("--probe input must be a JSON array of {role, content} objects")
            return 2
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        from typing import cast
        p_unsafe, top_token, cats = score_turn(
            client, args.model, cast(list[ChatCompletionMessageParam], msgs)
        )
        print(f"verdict      : {top_token}")
        print(f"P(unsafe)    : {p_unsafe:.6f}")
        print(f"categories   : {cats if cats else '—'}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)

    files = sorted(args.transcripts.glob("*.jsonl"))
    pending = [f for f in files if f.name not in done]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"found {len(files)} transcripts, {len(done)} already vetted, "
          f"{len(pending)} pending this run")
    if not pending:
        print_ranked(args.out, args.top)
        return 0

    # One client per worker thread; OpenAI() is thread-safe but we keep
    # connection pools warm by using a shared client.
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    t0 = time.time()
    n_done = 0
    n_unsafe = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(vet_transcript, f, client, args.model): f
                   for f in pending}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {"file": f.name, "verdict": "error", "error": str(e)}
            append_record(args.out, rec)
            n_done += 1
            if rec.get("verdict") == "unsafe":
                n_unsafe += 1
            if n_done % 25 == 0 or n_done == len(pending):
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(pending) - n_done) / max(rate, 1e-6)
                print(
                    f"  [{n_done}/{len(pending)}] {rate:.2f} files/s  "
                    f"eta {eta / 60:.1f}m  unsafe-so-far={n_unsafe}",
                    flush=True,
                )

    print(f"\nvet pass complete in {(time.time() - t0) / 60:.1f}m")
    print_ranked(args.out, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
