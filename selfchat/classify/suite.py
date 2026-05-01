"""Phase 5 separability suite — runs all four tiers and prints a unified
markdown table with CV / holdout metrics + interpretable features.

Usage (from project root):
  python -m selfchat.classify.suite
  python -m selfchat.classify.suite --seed freedom_dark
  python -m selfchat.classify.suite --seed freedom --transcripts transcripts/ \\
      --emb artifacts/emb_msgs.npz

Tier 3 reads its embedding artifact (--emb), which must already be filtered
or filterable to the chosen seed. The other tiers read transcript JSONLs
under --transcripts and filter by filename pattern.

If the requested seed is missing from the embedding artifact, Tier 3 is
skipped with a warning rather than failing the whole suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from selfchat.classify import tier0_length, tier1_chars, tier2_tfidf, tier3_embeddings
from selfchat.classify._common import DEFAULT_SEED, TierResult, format_table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help=f"Seed name to filter on (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--transcripts",
        type=Path,
        default=Path("transcripts"),
        help="Transcript directory for tiers 0–2 (default: transcripts/)",
    )
    p.add_argument(
        "--emb",
        type=Path,
        default=Path("artifacts/emb_msgs.npz"),
        help="Per-message embedding npz for tier 3 (default: artifacts/emb_msgs.npz)",
    )
    p.add_argument(
        "--k-features",
        type=int,
        default=3,
        help="How many top V/JB features to report per interpretable tier (default: 3)",
    )
    p.add_argument(
        "--skip-tier3",
        action="store_true",
        help="Skip tier 3 (e.g. when no current embedding artifact is available)",
    )
    args = p.parse_args()

    print(f"# Phase 5 separability suite — seed='{args.seed}'\n")

    results: list[TierResult] = []

    print("[1/4] Tier 0 — length+completion ...")
    results.extend(
        tier0_length.run(
            seed=args.seed,
            transcript_dir=args.transcripts,
            k_features=args.k_features,
        )
    )

    print("[2/4] Tier 1 — bag-of-characters ...")
    results.extend(
        tier1_chars.run(
            seed=args.seed,
            transcript_dir=args.transcripts,
            k_features=args.k_features,
        )
    )

    print("[3/4] Tier 2 — word TF-IDF ...")
    results.extend(
        tier2_tfidf.run(
            seed=args.seed,
            transcript_dir=args.transcripts,
            k_features=args.k_features,
        )
    )

    if args.skip_tier3:
        print("[4/4] Tier 3 — SKIPPED (--skip-tier3)")
    else:
        print("[4/4] Tier 3 — per-message embedding aggregations ...")
        try:
            results.extend(
                tier3_embeddings.run(seed=args.seed, emb_path=args.emb)
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ! tier 3 skipped: {e}")

    print()
    print(format_table(results, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
