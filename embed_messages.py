"""per-message nomic embeddings — Phase 3, orthogonal to embed.py.

Where embed.py produces ONE vector per run (terminal-state, last K turns
concatenated), this script produces ONE vector per individual message
(turn) of every natural-stop transcript. Output `emb_msgs.npz` is the
substrate for `probe.py` (turn-depth / agent decoding) and `plot_msgs.py`
(PCA-by-turn-index visualization).

Filters mirror embed.py for consistency:
  * legacy seed names (no longer in seeds.list_seed_names()) skipped
  * non-natural stops (interrupted / errored / in_flight) skipped
  * messages with zero word-regex matches dropped (punctuation-only slop)
  * per-message text truncated to DEFAULT_MAX_CHARS from the tail

Run: `python embed_messages.py [transcripts/] [--out emb_msgs.npz]`
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import BadRequestError, OpenAI

from analyze import NATURAL_STOPS, Transcript, is_current_seed, load
from embed import _has_words
from self_chat import OLLAMA_BASE_URL

DEFAULT_MODEL = "nomic-embed-text"
# Conservative initial cap. nomic-embed-text claims 2048 ctx but verbose
# model outputs (markdown / code / lists) can run dense; a per-call halving
# retry below handles overflows that slip through.
DEFAULT_MAX_CHARS = 3000
MIN_CHARS_AFTER_HALVING = 200

logger = logging.getLogger(__name__)


def embed_one_robust(text: str, model: str, client: OpenAI) -> tuple[list[float], int]:
    """Embed a single text. On context-overflow 400s, halve from the head
    (keeping the tail = terminal state) and retry until it fits or we hit
    MIN_CHARS_AFTER_HALVING. Returns (embedding, n_halvings)."""
    halvings = 0
    while True:
        try:
            resp = client.embeddings.create(input=[text], model=model)
            return resp.data[0].embedding, halvings
        except BadRequestError as e:
            msg = str(e).lower()
            if "context length" not in msg and "exceeds" not in msg:
                raise
            if len(text) <= MIN_CHARS_AFTER_HALVING:
                raise
            text = text[len(text) // 2 :]
            halvings += 1


def embed_texts_robust(
    texts: list[str], model: str, client: OpenAI
) -> tuple[np.ndarray, int]:
    """Embed texts one-at-a-time with halving-retry. Returns (vectors, n_halved)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), 0
    vecs: list[list[float]] = []
    n_halved = 0
    for i, text in enumerate(texts):
        emb, halvings = embed_one_robust(text, model, client)
        vecs.append(emb)
        if halvings:
            n_halved += 1
        if (i + 1) % 250 == 0 or i + 1 == len(texts):
            logger.info("embedded %d/%d (%d halved)", i + 1, len(texts), n_halved)
    return np.asarray(vecs, dtype=np.float32), n_halved


def gather_messages(transcripts: list[Transcript]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Flatten transcripts into per-message rows, dropping word-empty messages."""
    rows: list[dict] = []
    for t in transcripts:
        for turn in t.turns:
            if not _has_words(turn.content):
                continue
            content = turn.content
            if len(content) > DEFAULT_MAX_CHARS:
                content = content[-DEFAULT_MAX_CHARS:]
            rows.append(
                {
                    "text": content,
                    "run_id": t.run_id,
                    "turn_index": turn.turn_index,
                    "agent": turn.agent,
                    "variant": t.model_variant,
                    "seed": t.seed_name,
                    "stop_reason": t.stop_reason,
                    "completed_turns": t.completed_turns,
                }
            )
    texts = [r["text"] for r in rows]
    meta = {
        "run_ids": np.array([r["run_id"] for r in rows]),
        "turn_index": np.array([r["turn_index"] for r in rows], dtype=np.int32),
        "agents": np.array([r["agent"] for r in rows]),
        "variants": np.array([r["variant"] for r in rows]),
        "seeds": np.array([r["seed"] for r in rows]),
        "stop_reasons": np.array([r["stop_reason"] for r in rows]),
        "completed_turns": np.array([r["completed_turns"] for r in rows], dtype=np.int32),
    }
    return texts, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dir",
        type=Path,
        nargs="?",
        default=Path("transcripts"),
        help="Directory of *.jsonl transcript files (default: transcripts/)",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Ollama embed model (default: {DEFAULT_MODEL})"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("emb_msgs.npz"),
        help="Output .npz path (default: emb_msgs.npz)",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Embed at most N runs; 0 = all (useful for smoke tests)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not args.dir.exists():
        print(f"not found: {args.dir}")
        return 2

    paths = sorted(args.dir.glob("*.jsonl"))
    print(f"found {len(paths)} transcript file(s) in {args.dir}")

    transcripts: list[Transcript] = []
    legacy = aborted = empty = 0
    for path in paths:
        t = load(path)
        if t is None:
            continue
        if not is_current_seed(t.seed_name):
            legacy += 1
            continue
        if t.stop_reason not in NATURAL_STOPS:
            aborted += 1
            continue
        if not t.turns:
            empty += 1
            continue
        transcripts.append(t)

    if legacy:
        print(f"  ({legacy} skipped: legacy seed)")
    if aborted:
        print(f"  ({aborted} skipped: aborted/in-flight)")
    if empty:
        print(f"  ({empty} skipped: no turns)")
    print(f"  {len(transcripts)} natural-stop transcripts retained")

    if args.max_runs > 0:
        transcripts = transcripts[: args.max_runs]
        print(f"  --max-runs={args.max_runs}: subset to {len(transcripts)} runs")

    texts, meta = gather_messages(transcripts)
    print(f"  → {len(texts)} messages to embed")

    if not texts:
        print("nothing to embed.")
        return 1

    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for v, s in zip(meta["variants"], meta["seeds"], strict=True):
        counts[(str(v), str(s))] += 1
    print()
    print(f"{'variant':<11}  {'seed':<22}  {'n_msgs':>7}")
    print("-" * 46)
    for (v, s), n in sorted(counts.items()):
        print(f"{v:<11}  {s:<22}  {n:>7}")
    print()

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    try:
        vecs, n_halved = embed_texts_robust(texts, model=args.model, client=client)
    except Exception as e:
        print(f"embedding failed: {e}")
        print(f"hint: did you `ollama pull {args.model}`?")
        return 1
    print(f"embeddings shape: {vecs.shape}")
    if n_halved:
        print(f"  ({n_halved} message(s) needed halving-retry to fit nomic context)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        vectors=vecs,
        model=args.model,
        run_ids=meta["run_ids"],
        turn_index=meta["turn_index"],
        agents=meta["agents"],
        variants=meta["variants"],
        seeds=meta["seeds"],
        stop_reasons=meta["stop_reasons"],
        completed_turns=meta["completed_turns"],
    )
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
