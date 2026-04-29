"""terminal-state embeddings via an Ollama-served embedding model.

For each transcript, takes the last K turns concatenated as the 'terminal
state', embeds via Ollama's OpenAI-compat embeddings API, and prints a
cohesion/separation summary plus optionally writes a .npz of vectors + metadata.

  cohesion (per cell) = mean pairwise cosine similarity WITHIN that cell
  separation          = mean pairwise cosine similarity ACROSS different cells

A larger cohesion-vs-separation gap means terminal states cluster by cell —
direct evidence of cell-distinct attractors. Aborted runs are excluded.

Run: `python embed.py [transcripts/] [--model nomic-embed-text] [--last-k 5] [--out emb.npz]`

The embedding model must be pulled into Ollama first, e.g.:
  ollama pull nomic-embed-text
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI

from analyze import NATURAL_STOPS, Transcript, Turn, is_current_seed, load
from self_chat import OLLAMA_BASE_URL

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_LAST_K = 5
DEFAULT_MAX_CHARS = 4000  # ~1000 tokens, well under nomic-embed-text's 2048 context

# Shared with browse.py's TF-IDF tooltip tokenizer so the embedding pipeline
# and the displayed top-tokens agree on what counts as a "word."
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

logger = logging.getLogger(__name__)


def _has_words(s: str) -> bool:
    """True iff `s` contains at least one alphabetic word (per WORD_RE)."""
    return bool(WORD_RE.search(s))


def _collapse_trailing_repeats(turns: list[Turn]) -> list[Turn]:
    """Collapse a trailing run of identical-content turns to its first occurrence.

    Degenerate-repetition runs end with K-of-a-kind identical messages (often
    just "." or other low-content strings). Taking the literal last K turns
    for embedding then yields a string of K copies of that token — pure noise.
    Replacing the run with a single representative (the FIRST occurrence —
    where the loop began) preserves the existence-of-loop signal while
    reclaiming the other K-1 slots for the actually-distinct turns that
    preceded it.

    No-op for transcripts without trailing repetition (e.g. completed runs).
    """
    if len(turns) < 2:
        return turns
    last = turns[-1].content
    i = len(turns) - 1
    while i > 0 and turns[i - 1].content == last:
        i -= 1
    if i == len(turns) - 1:
        return turns
    return turns[: i + 1]


def terminal_state(t: Transcript, last_k: int, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Concatenate last K word-bearing turn contents; truncate to max_chars (keep tail).

    Two filters run before the last-K slice, both addressing the same confound
    (terminal-state contamination by degenerate slop):

      1. Drop turns with zero word-regex matches. Punctuation-only or
         whitespace-only outputs (e.g. "( . . . )") have no embeddable
         lexical content; they spread along monotone progressions that
         elude byte-equality collapse and dominate last-K when included.
         The "loop existed" signal is preserved by stop_reason metadata,
         not by encoding slop in the embedding.
      2. Collapse trailing byte-identical turns to one occurrence — handles
         the orthogonal failure mode of word-bearing repetition loops
         (e.g. K turns of "I love you.") that wouldn't be caught by (1).
    """
    if not t.turns:
        return ""
    turns = [turn for turn in t.turns if _has_words(turn.content)]
    if not turns:
        return ""
    turns = _collapse_trailing_repeats(turns)
    tail = turns[-last_k:] if last_k > 0 else turns
    text = "\n".join(turn.content for turn in tail)
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]  # keep the tail (the actual terminal state)
    return text


def embed_texts(texts: list[str], model: str, client: OpenAI) -> np.ndarray:
    """Embed texts one at a time. Returns (N, D) float32 array.

    One-at-a-time avoids batch-size context limits in some Ollama embedding
    backends (e.g. nomic-embed-text, which 400s on multi-input batches that
    exceed its 2048-token context).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs: list[list[float]] = []
    for i, text in enumerate(texts):
        resp = client.embeddings.create(input=[text], model=model)
        vecs.append(resp.data[0].embedding)
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            logger.info("embedded %d/%d", i + 1, len(texts))
    return np.asarray(vecs, dtype=np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def cohesion_separation(
    vecs: np.ndarray, cell_ids: np.ndarray
) -> tuple[dict[int, float], float, np.ndarray]:
    """Mean pairwise cosine sim within each cell + mean across-cell pairwise.

    Returns (per-cell-cohesion, separation, sim_matrix).
    sim_matrix is (N, N) cosine sims for downstream inspection.
    """
    if len(vecs) == 0:
        return {}, 0.0, np.zeros((0, 0))
    normed = _l2_normalize(vecs)
    sim = normed @ normed.T
    n = len(vecs)
    same_cell = cell_ids[:, None] == cell_ids[None, :]
    off_diag = ~np.eye(n, dtype=bool)

    cohesion: dict[int, float] = {}
    for cid in np.unique(cell_ids):
        mask = (cell_ids == cid)[:, None] & (cell_ids == cid)[None, :] & off_diag
        if mask.sum() == 0:
            cohesion[int(cid)] = float("nan")
        else:
            cohesion[int(cid)] = float(sim[mask].mean())

    cross_mask = (~same_cell) & off_diag
    separation = float(sim[cross_mask].mean()) if cross_mask.any() else float("nan")
    return cohesion, separation, sim


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
        "--last-k",
        type=int,
        default=DEFAULT_LAST_K,
        help=f"Last K turns as terminal state (default: {DEFAULT_LAST_K})",
    )
    p.add_argument("--out", type=Path, help="Optional .npz path to save vectors + metadata")
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
    legacy = 0
    for path in paths:
        t = load(path)
        if t is None:
            continue
        if not is_current_seed(t.seed_name):
            legacy += 1
            continue
        if t.stop_reason not in NATURAL_STOPS:
            continue  # exclude in_flight / interrupted / errored
        if not t.turns:
            continue
        transcripts.append(t)

    if legacy:
        print(f"  ({legacy} file(s) skipped: legacy seed name)")

    if not transcripts:
        print("no natural-stop transcripts to embed")
        return 1
    print(
        f"embedding {len(transcripts)} natural-stop transcripts "
        f"(last_k={args.last_k}, model={args.model})"
    )

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    texts = [terminal_state(t, args.last_k) for t in transcripts]

    try:
        vecs = embed_texts(texts, model=args.model, client=client)
    except Exception as e:
        print(f"embedding failed: {e}")
        print(f"hint: did you `ollama pull {args.model}`?")
        return 1
    print(f"embeddings shape: {vecs.shape}")

    variants = np.array([t.model_variant for t in transcripts])
    seeds = np.array([t.seed_name for t in transcripts])
    run_ids = np.array([t.run_id for t in transcripts])
    stop_reasons = np.array([t.stop_reason for t in transcripts])
    cell_labels = np.array([f"{v}|{s}" for v, s in zip(variants, seeds, strict=True)])
    cells_unique, cell_idx = np.unique(cell_labels, return_inverse=True)

    cohesion, separation, _sim = cohesion_separation(vecs, cell_idx)

    print()
    print(f"{'variant':<11}  {'seed':<12}  {'n':>3}  {'cohesion':>9}")
    print("-" * 42)
    counts: defaultdict[int, int] = defaultdict(int)
    for i in cell_idx:
        counts[int(i)] += 1
    for ci, label in enumerate(cells_unique):
        v, s = label.split("|", 1)
        coh = cohesion.get(ci, float("nan"))
        coh_str = f"{coh:.3f}" if not np.isnan(coh) else "  n/a"
        print(f"{v:<11}  {s:<12}  {counts[ci]:>3}  {coh_str:>9}")
    print("-" * 42)
    sep_str = f"{separation:.3f}" if not np.isnan(separation) else "n/a"
    print(f"  separation (across cells) : {sep_str}")
    coh_vals = [v for v in cohesion.values() if not np.isnan(v)]
    if coh_vals:
        mean_coh = float(np.mean(coh_vals))
        print(f"  mean cohesion (within)    : {mean_coh:.3f}")
        print(f"  contrast (cohesion - sep) : {mean_coh - separation:+.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out,
            vectors=vecs,
            run_ids=run_ids,
            variants=variants,
            seeds=seeds,
            stop_reasons=stop_reasons,
            last_k=args.last_k,
            model=args.model,
        )
        print(f"\nsaved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
