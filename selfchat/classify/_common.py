"""Shared helpers for the Tier 0–3 separability classifiers + suite runner."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_SEED = "freedom"


def make_seed_re(seed: str) -> re.Pattern[str]:
    """Match transcript filenames `<variant>_<seed>_<32hex>_<timestamp>.jsonl`.

    The 32-hex run_id anchor cleanly excludes seeds that share a prefix
    (e.g. seed='freedom' will not match `freedom_dark_<id>_*.jsonl`,
    because after `freedom_` the regex requires 32 hex chars and `dark_`
    fails on `r`).
    """
    return re.compile(
        rf"^(?P<variant>vanilla|jailbroken)_{re.escape(seed)}_(?P<run_id>[0-9a-f]{{32}})_"
    )


@dataclass
class TierResult:
    """One row of the separability suite table.

    Each tier-script's `run()` function returns one or more of these.
    """

    tier: str        # "0", "1", "2", "3"
    mode: str        # "length", "char (1,2)", "tfidf (1,1)", "emb mean", ...
    n: int           # n_runs by default; for mode-D msg-level pass n_msgs
    cv_acc_mean: float
    cv_acc_std: float
    cv_auc_mean: float
    cv_auc_std: float
    holdout_acc: float | None = None
    holdout_auc: float | None = None
    # Interpretable features. Empty for tiers without per-feature coefficients
    # (e.g. Tier 3 embedding dims). Populated from a full-data logreg fit.
    top_v_features: list[str] = field(default_factory=list)
    top_jb_features: list[str] = field(default_factory=list)


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _fmt_holdout(val: float | None, digits: int = 3) -> str:
    return f"{val:.{digits}f}" if val is not None else "—"


def _fmt_features(feats: list[str], k: int = 3) -> str:
    if not feats:
        return "—"
    return ", ".join(feats[:k])


def format_table(results: list[TierResult], seed: str) -> str:
    """Render results as a markdown table. Header line includes seed."""
    header = (
        f"## Phase 5 separability suite — seed `{seed}` "
        f"(5×5 repeated CV)\n\n"
    )
    cols = (
        "| Tier | Mode | n | CV acc | CV AUC | "
        "Holdout acc | Holdout AUC | V-affine (top) | JB-affine (top) |\n"
    )
    sep = (
        "|---|---|---:|---:|---:|---:|---:|---|---|\n"
    )
    rows: list[str] = []
    for r in results:
        rows.append(
            f"| **{r.tier}** | {r.mode} | {r.n} | "
            f"{_fmt_pm(r.cv_acc_mean, r.cv_acc_std)} | "
            f"{_fmt_pm(r.cv_auc_mean, r.cv_auc_std)} | "
            f"{_fmt_holdout(r.holdout_acc)} | "
            f"{_fmt_holdout(r.holdout_auc)} | "
            f"{_fmt_features(r.top_v_features)} | "
            f"{_fmt_features(r.top_jb_features)} |"
        )
    return header + cols + sep + "\n".join(rows) + "\n"
