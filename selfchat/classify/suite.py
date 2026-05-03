"""Phase 5 separability suite — runs all four tiers across one or more seeds
and prints a unified markdown table per seed plus a master comparison matrix.
Also writes a JSON cache of `TierResult`s consumed by
`selfchat.viz.plot_separability`.

Usage (from project root):
  python -m selfchat.classify.suite                              # all 7 default seeds
  python -m selfchat.classify.suite --seeds freedom              # one seed
  python -m selfchat.classify.suite --seeds freedom freedom_dark # subset
  python -m selfchat.classify.suite --skip-tier3                 # skip embedding tier

Tier 3 reads its embedding artifact (--emb), which must already be filtered
or filterable to the chosen seed. The other tiers read transcript JSONLs
under --transcripts and filter by filename pattern. If the requested seed
is missing from the embedding artifact, Tier 3 is skipped with a warning
rather than failing the whole run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from selfchat.classify import tier0_length, tier1_chars, tier2_tfidf, tier3_embeddings
from selfchat.classify._common import DEFAULT_SEED, TierResult, format_table

DEFAULT_SEEDS = (
    "freedom",
    "freedom_dark",
    "freedom_neg_minimal",
    "escaped",
    "unbound",
    "task",
    "task_free",
)


def run_seed(
    seed: str,
    transcript_dir: Path,
    emb_path: Path,
    k_features: int,
    skip_tier3: bool,
) -> list[TierResult]:
    out: list[TierResult] = []
    out.extend(tier0_length.run(seed=seed, transcript_dir=transcript_dir,
                                k_features=k_features))
    out.extend(tier1_chars.run(seed=seed, transcript_dir=transcript_dir,
                               k_features=k_features))
    out.extend(tier2_tfidf.run(seed=seed, transcript_dir=transcript_dir,
                               k_features=k_features))
    if not skip_tier3:
        try:
            out.extend(tier3_embeddings.run(seed=seed, emb_path=emb_path))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ! tier 3 skipped for seed='{seed}': {e}")
    return out


def _row(seed: str, results: list[TierResult]) -> dict[str, float | str]:
    """Pull the one headline AUC per tier for the master matrix."""
    by_mode = {(r.tier, r.mode): r for r in results}
    pick = {
        "tier 0": by_mode.get(("0", "length+completion")),
        "tier 1 char(1,2)": by_mode.get(("1", "char (1,2)")),
        "tier 2 tfidf(1,2)": by_mode.get(("2", "tfidf (1,2)")),
        "tier 3C": by_mode.get(("3", "emb mean‖std (C)")),
        "tier 3D run": by_mode.get(("3", "emb per-msg (D, run-level)")),
        "tier 3D msg": by_mode.get(("3", "emb per-msg (D, msg-level)")),
    }
    row: dict[str, float | str] = {"seed": seed}
    for k, r in pick.items():
        row[k] = f"{r.cv_auc_mean:.3f}" if r is not None else "—"
    n_runs = next((r.n for r in results if r.tier == "0"), None)
    row["n"] = str(n_runs) if n_runs is not None else "—"
    return row


def format_master_matrix(rows: list[dict[str, float | str]]) -> str:
    cols = ["seed", "n", "tier 0", "tier 1 char(1,2)", "tier 2 tfidf(1,2)",
            "tier 3C", "tier 3D run", "tier 3D msg"]
    header = "## Master matrix — CV AUC by seed × tier\n\n"
    sep_left = "|---" + "|---:" * (len(cols) - 1) + "|"
    line = "| " + " | ".join(cols) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |")
    return header + line + "\n" + sep_left + "\n" + "\n".join(body) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds", nargs="+", default=list(DEFAULT_SEEDS),
        help=f"One or more seed names to run (default: all 7 — {' '.join(DEFAULT_SEEDS)})",
    )
    p.add_argument(
        "--seed", dest="seeds", nargs=1, default=argparse.SUPPRESS,
        help=f"Backward-compat alias for --seeds (single value, default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--transcripts", type=Path, default=Path("transcripts"),
        help="Transcript directory for tiers 0–2 (default: transcripts/)",
    )
    p.add_argument(
        "--emb", type=Path, default=Path("artifacts/emb_msgs.npz"),
        help="Per-message embedding npz for tier 3 (default: artifacts/emb_msgs.npz)",
    )
    p.add_argument(
        "--k-features", type=int, default=3,
        help="Top V/JB features per interpretable tier (default: 3)",
    )
    p.add_argument(
        "--skip-tier3", action="store_true",
        help="Skip tier 3 (e.g. when no embedding artifact is available)",
    )
    p.add_argument(
        "--cache", type=Path,
        default=Path("artifacts/separability_results.json"),
        help="Where to write the JSON cache consumed by "
             "selfchat.viz.plot_separability "
             "(default: artifacts/separability_results.json)",
    )
    args = p.parse_args()

    print(f"# Phase 5 separability suite — {len(args.seeds)} seed(s) × 4 tiers\n")
    t0 = time.time()

    all_results: dict[str, list[TierResult]] = {}
    for i, seed in enumerate(args.seeds, 1):
        print(f"[{i}/{len(args.seeds)}] seed='{seed}' ...")
        t_seed = time.time()
        results = run_seed(seed, args.transcripts, args.emb,
                           args.k_features, args.skip_tier3)
        all_results[seed] = results
        print(f"  ({time.time() - t_seed:.1f}s, {len(results)} rows)")

    print(f"\nsuite complete in {time.time() - t0:.1f}s\n")

    for seed, results in all_results.items():
        print(format_table(results, seed))
        print()

    rows = [_row(seed, results) for seed, results in all_results.items()]
    print(format_master_matrix(rows))

    serial = {
        seed: {tr.mode: asdict(tr) for tr in results}
        for seed, results in all_results.items()
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(serial, indent=2))
    print(f"cached: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
