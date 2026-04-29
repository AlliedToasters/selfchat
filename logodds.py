"""log-odds with informative Dirichlet prior.

Ranks tokens (and optionally bigrams) by how strongly they distinguish two
variants on a shared corpus (e.g. one seed). Method follows Monroe, Colaresi,
Quinn 2008 ("Fightin' Words"): smoothed log-odds against a pooled prior,
then z-scored by the asymptotic variance estimate.

For each token w in the union vocabulary,

    pooled[w] = count_jb[w] + count_va[w]                  # informative prior
    omega_i[w]   = log( (y_i[w] + pooled[w])
                       / (n_i + alpha_0 - y_i[w] - pooled[w]) )
    delta[w]     = omega_jb[w] - omega_va[w]
    var(delta)   = 1 / (y_jb[w] + pooled[w]) + 1 / (y_va[w] + pooled[w])
    z[w]         = delta[w] / sqrt(var(delta[w]))

Positive z = more characteristic of jailbroken; negative = vanilla.

Run:
    python logodds.py --seed escape_lead
    python logodds.py --seed escape_lead --turn-range 0:5
    python logodds.py --seed escape_lead --bigrams --csv ranked.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from analyze import NATURAL_STOPS, Transcript, is_current_seed, load
from embed import WORD_RE
from shared import _STOPWORDS

def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in WORD_RE.findall(text) if tok.lower() not in _STOPWORDS]

def _bigrams(tokens: list[str]) -> Iterable[str]:
    return (f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))


def collect_tokens(t: Transcript, turn_range: slice, include_bigrams: bool) -> list[str]:
    """Tokenize the selected turn-range of a transcript; optionally append bigrams."""
    selected = t.turns[turn_range]
    out: list[str] = []
    for turn in selected:
        toks = tokenize(turn.content)
        out.extend(toks)
        if include_bigrams:
            out.extend(_bigrams(toks))
    return out


def log_odds(
    counts_a: Counter[str],
    counts_b: Counter[str],
    min_count: int = 5,
) -> list[tuple[str, float, int, int]]:
    """Informative-Dirichlet log-odds z-scores for A-vs-B.

    Returns a list of (token, z, count_a, count_b) sorted by z descending.
    Positive z = more characteristic of A; negative = more characteristic of B.
    Tokens whose pooled count is below `min_count` are dropped to suppress
    rare-word noise.
    """
    pooled: Counter[str] = counts_a + counts_b
    vocab = [w for w, c in pooled.items() if c >= min_count]
    if not vocab:
        return []

    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    alpha_0 = sum(pooled.values())

    out: list[tuple[str, float, int, int]] = []
    for w in vocab:
        y_a = counts_a.get(w, 0)
        y_b = counts_b.get(w, 0)
        a_w = pooled[w]
        # Smoothed log-odds in each corpus against the pooled prior.
        omega_a = math.log((y_a + a_w) / (n_a + alpha_0 - y_a - a_w))
        omega_b = math.log((y_b + a_w) / (n_b + alpha_0 - y_b - a_w))
        delta = omega_a - omega_b
        var = 1.0 / (y_a + a_w) + 1.0 / (y_b + a_w)
        z = delta / math.sqrt(var)
        out.append((w, z, y_a, y_b))
    out.sort(key=lambda r: r[1], reverse=True)
    return out


def _parse_turn_range(s: str) -> slice:
    """Parse 'a:b' / ':b' / 'a:' / 'all' into a slice."""
    if s == "all":
        return slice(None)
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"turn range must be 'a:b', ':b', 'a:', or 'all' (got {s!r})"
        )
    a_s, b_s = s.split(":", 1)
    a = int(a_s) if a_s else None
    b = int(b_s) if b_s else None
    return slice(a, b)


def _print_ranked(
    ranked: list[tuple[str, float, int, int]], top: int, label_a: str, label_b: str
) -> None:
    if not ranked:
        print("(no tokens above min-count)")
        return
    head = f"  {'token':<30} {'z':>7} {label_a[:5]:>5} {label_b[:5]:>5}"

    print()
    print(f"top {top} {label_a}-leaning (positive z):")
    print(head)
    for w, z, y_a, y_b in ranked[:top]:
        print(f"  {w:<30} {z:>7.2f} {y_a:>5} {y_b:>5}")

    print()
    print(f"top {top} {label_b}-leaning (negative z):")
    print(head)
    for w, z, y_a, y_b in ranked[-top:][::-1]:
        print(f"  {w:<30} {z:>7.2f} {y_a:>5} {y_b:>5}")


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
        "--seed",
        type=str,
        default="all",
        help="Filter to a single seed name (default: 'all' = pool current seeds)",
    )
    p.add_argument(
        "--turn-range",
        type=_parse_turn_range,
        default=slice(None),
        metavar="A:B",
        help="Slice of turns to analyze, e.g. '0:5', ':10', '5:', 'all' (default: all)",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Drop tokens with fewer total occurrences across both variants (default: 5)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top tokens to print on each side (default: 30)",
    )
    p.add_argument(
        "--bigrams",
        action="store_true",
        help="Also score bigrams (more vocab, slower; off by default)",
    )
    p.add_argument(
        "--include-aborted",
        action="store_true",
        help="Include non-natural-stop runs (default: natural-only, matching markers.py)",
    )
    p.add_argument("--csv", type=Path, help="Write full ranked CSV to this path")
    args = p.parse_args()

    if not args.dir.exists():
        print(f"not found: {args.dir}")
        return 2

    paths = sorted(args.dir.glob("*.jsonl"))
    counts: dict[str, Counter[str]] = {"vanilla": Counter(), "jailbroken": Counter()}
    n_runs: dict[str, int] = {"vanilla": 0, "jailbroken": 0}
    skipped_legacy = 0
    for path in paths:
        t = load(path)
        if t is None:
            continue
        if not is_current_seed(t.seed_name):
            skipped_legacy += 1
            continue
        if args.seed != "all" and t.seed_name != args.seed:
            continue
        if not args.include_aborted and t.stop_reason not in NATURAL_STOPS:
            continue
        if t.model_variant not in counts:
            continue
        toks = collect_tokens(t, args.turn_range, args.bigrams)
        counts[t.model_variant].update(toks)
        n_runs[t.model_variant] += 1

    print(f"found {len(paths)} transcript file(s) in {args.dir}")
    if skipped_legacy:
        print(f"  ({skipped_legacy} skipped: legacy seed name)")
    rng = args.turn_range
    if rng == slice(None):
        rng_str = "all"
    else:
        rng_str = f"{rng.start or 0}:{rng.stop if rng.stop is not None else ''}"
    print(
        f"seed={args.seed}  turn_range={rng_str}  bigrams={args.bigrams}  "
        f"natural_only={not args.include_aborted}  min_count={args.min_count}"
    )
    n_va_tokens = sum(counts["vanilla"].values())
    n_jb_tokens = sum(counts["jailbroken"].values())
    print(f"vanilla:    n_runs={n_runs['vanilla']:>3}  n_tokens={n_va_tokens}")
    print(f"jailbroken: n_runs={n_runs['jailbroken']:>3}  n_tokens={n_jb_tokens}")

    if not n_runs["vanilla"] or not n_runs["jailbroken"]:
        print("\n(insufficient data — both variants must have transcripts)")
        return 1

    ranked = log_odds(counts["jailbroken"], counts["vanilla"], min_count=args.min_count)
    _print_ranked(ranked, top=args.top, label_a="jailbroken", label_b="vanilla")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["token", "z", "count_jailbroken", "count_vanilla"])
            for tok, z, y_jb, y_va in ranked:
                w.writerow([tok, f"{z:.4f}", y_jb, y_va])
        print(f"\nfull ranked CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
