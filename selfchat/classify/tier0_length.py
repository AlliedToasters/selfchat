"""Tier 0 null-hypothesis test: can we separate vanilla vs. jailbroken
self-chat rollouts on the `freedom` seed using only message-length and
completion features? No content, no embeddings, no tokens.

The `freedom` seed has no harm pressure — it's the same prompt Anthropic's
Opus 4 system card used. If V/JB rollouts are NOT separable here, the
Phase 4 manipulation-basin finding is task-conditional (refusal-relevant
priming required). If they ARE separable, abliteration has shifted the
model's distribution beyond just refusal-relevant content.

Per-transcript features:
  * completed_turns       (int 1..50)
  * is_early_stop         (bool: completed_turns < 50)
  * stop_reason_degenerate (1 if degenerate_repetition else 0)
  * total_chars           (sum of all turn content)
  * mean_msg_chars        (per turn average)
  * median_msg_chars
  * std_msg_chars
  * max_msg_chars
  * min_msg_chars

Run:
  python classify_freedom_tier0.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

from selfchat.analysis.analyze import NATURAL_STOPS, load as load_transcript
from selfchat.classify._common import DEFAULT_SEED, TierResult, make_seed_re

TRANSCRIPT_DIR = Path("transcripts")

FEATURES = [
    "completed_turns",
    "is_early_stop",
    "stop_reason_degenerate",
    "total_chars",
    "mean_msg_chars",
    "median_msg_chars",
    "std_msg_chars",
    "max_msg_chars",
    "min_msg_chars",
]


def extract_features(
    path: Path, seed_re: re.Pattern[str]
) -> tuple[str, dict[str, float]] | None:
    """Return (variant, features) for one transcript, or None if malformed
    or in-flight. Uses `analyze.load()` for robust JSONL parsing."""
    m = seed_re.match(path.name)
    if m is None:
        return None
    variant = m.group("variant")

    t = load_transcript(path)
    if t is None or t.stop_reason not in NATURAL_STOPS or not t.turns:
        return None

    contents = [turn.content for turn in t.turns]
    lengths = np.array([len(c) for c in contents], dtype=np.float64)
    feats: dict[str, float] = {
        "completed_turns": float(t.completed_turns),
        "is_early_stop": float(t.completed_turns < 50),
        "stop_reason_degenerate": float(t.stop_reason == "degenerate_repetition"),
        "total_chars": float(lengths.sum()),
        "mean_msg_chars": float(lengths.mean()),
        "median_msg_chars": float(np.median(lengths)),
        "std_msg_chars": float(lengths.std()),
        "max_msg_chars": float(lengths.max()),
        "min_msg_chars": float(lengths.min()),
    }
    return variant, feats


def collect(
    seed: str = DEFAULT_SEED, transcript_dir: Path = TRANSCRIPT_DIR
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    seed_re = make_seed_re(seed)
    rows: list[tuple[str, list[float]]] = []
    for p in sorted(transcript_dir.iterdir()):
        result = extract_features(p, seed_re)
        if result is None:
            continue
        variant, feats = result
        rows.append((variant, [feats[k] for k in FEATURES]))

    if not rows:
        raise RuntimeError(
            f"no transcripts matching seed='{seed}' under {transcript_dir}/"
        )

    variants = np.array([r[0] for r in rows])
    X = np.array([r[1] for r in rows], dtype=np.float64)
    return X, variants, FEATURES


def describe_by_variant(X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    print(f"  {'feature':<24}  {'V mean':>10}  {'JB mean':>10}  "
          f"{'V med':>10}  {'JB med':>10}  {'Δmean/σ':>8}")
    for i, name in enumerate(names):
        v_vals = X[y == "vanilla", i]
        j_vals = X[y == "jailbroken", i]
        v_mu, v_med = v_vals.mean(), float(np.median(v_vals))
        j_mu, j_med = j_vals.mean(), float(np.median(j_vals))
        # pooled standardized mean diff (Cohen's d-ish)
        n_v, n_j = len(v_vals), len(j_vals)
        s2 = ((n_v - 1) * v_vals.var(ddof=1) + (n_j - 1) * j_vals.var(ddof=1)) / max(
            1, n_v + n_j - 2
        )
        s = float(np.sqrt(s2)) if s2 > 0 else float("nan")
        d = (j_mu - v_mu) / s if s and not np.isnan(s) else float("nan")
        print(
            f"  {name:<24}  {v_mu:>10.1f}  {j_mu:>10.1f}  "
            f"{v_med:>10.1f}  {j_med:>10.1f}  {d:>+8.2f}"
        )


def univariate_threshold(
    X: np.ndarray, y: np.ndarray, names: list[str]
) -> None:
    """For each feature, find the best threshold and report accuracy.
    Sweeps every value as a candidate threshold and picks the best."""
    y_bin = (y == "jailbroken").astype(int)
    print(f"  {'feature':<24}  {'best thr':>10}  {'direction':>10}  "
          f"{'acc':>6}  {'auc':>6}")
    for i, name in enumerate(names):
        vals = X[:, i]
        best_acc = 0.0
        best_thr = float("nan")
        best_dir = ""
        # Try both directions (>thr → JB, and <thr → JB). Sweep every
        # midpoint between sorted unique values.
        uniq = np.unique(vals)
        cand = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else uniq
        for thr in cand:
            for direction, pred_thr in (
                (">", (vals > thr).astype(int)),
                ("<", (vals < thr).astype(int)),
            ):
                acc = float((pred_thr == y_bin).mean())
                if acc > best_acc:
                    best_acc, best_thr, best_dir = acc, float(thr), direction
        try:
            auc = roc_auc_score(y_bin, vals)
            if auc < 0.5:
                auc = 1 - auc  # report direction-agnostic AUC
        except Exception:
            auc = float("nan")
        print(
            f"  {name:<24}  {best_thr:>10.2f}  {best_dir:>10}  "
            f"{best_acc:>6.3f}  {auc:>6.3f}"
        )


def cv_logreg_stats(
    X: np.ndarray, y: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cv_accs, cv_aucs) under 5×5 repeated stratified CV."""
    y_bin = (y == "jailbroken").astype(int)
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(X, y_bin):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(
            scaler.transform(X[tr]), y_bin[tr]
        )
        proba = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y_bin[te], pred))
        aucs.append(roc_auc_score(y_bin[te], proba))
    return np.array(accs), np.array(aucs)


def full_data_coefs(
    X: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0
) -> list[tuple[str, float]]:
    """Fit logreg on standardized full data, return [(feature, coef), ...]
    sorted by signed coefficient (most negative = most V-affine first)."""
    y_bin = (y == "jailbroken").astype(int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(
        scaler.transform(X), y_bin
    )
    coefs = clf.coef_.ravel()
    return sorted(zip(names, coefs.tolist(), strict=True), key=lambda t: t[1])


def cv_logreg(X: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0) -> None:
    accs, aucs = cv_logreg_stats(X, y, seed)
    print(
        f"  5×5 repeated CV: acc = {accs.mean():.3f} ± {accs.std():.3f}  "
        f"|  AUC = {aucs.mean():.3f} ± {aucs.std():.3f}"
    )
    print("\n  full-data coefficients (standardized features; +→ JB, −→ V):")
    pairs = full_data_coefs(X, y, names, seed)
    pairs_by_abs = sorted(pairs, key=lambda t: -abs(t[1]))
    for name, coef in pairs_by_abs:
        print(f"    {name:<24}  {coef:+.3f}")


def held_out_stats(
    X: np.ndarray, y: np.ndarray, seed: int = 0
) -> tuple[float, float, np.ndarray]:
    """One 70/30 stratified split. Returns (acc, auc, confusion_matrix)."""
    y_bin = (y == "jailbroken").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_bin, test_size=0.3, stratify=y_bin, random_state=seed
    )
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(
        scaler.transform(Xtr), ytr
    )
    proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
    pred = (proba >= 0.5).astype(int)
    return (
        float(accuracy_score(yte, pred)),
        float(roc_auc_score(yte, proba)),
        confusion_matrix(yte, pred),
    )


def held_out_split(X: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
    acc, auc, cm = held_out_stats(X, y, seed)
    n_test = int(cm.sum())
    print(f"  70/30 holdout (seed={seed}): n_test={n_test}")
    print(f"  acc = {acc:.3f}  |  AUC = {auc:.3f}")
    print("  confusion matrix [rows=true V/JB, cols=pred V/JB]:")
    print(f"    [[{cm[0,0]:>3} {cm[0,1]:>3}]")
    print(f"     [{cm[1,0]:>3} {cm[1,1]:>3}]]")


def run(
    seed: str = DEFAULT_SEED, transcript_dir: Path = TRANSCRIPT_DIR, k_features: int = 3
) -> list[TierResult]:
    """Programmatic entry point used by the suite runner.

    Returns one TierResult for the all-features logreg.
    """
    X, y, names = collect(seed=seed, transcript_dir=transcript_dir)
    accs, aucs = cv_logreg_stats(X, y)
    h_acc, h_auc, _ = held_out_stats(X, y)
    pairs = full_data_coefs(X, y, names)  # sorted ascending by signed coef
    top_v = [name for name, _ in pairs[:k_features]]
    top_jb = [name for name, _ in reversed(pairs[-k_features:])]
    return [
        TierResult(
            tier="0",
            mode="length+completion",
            n=len(y),
            cv_acc_mean=float(accs.mean()),
            cv_acc_std=float(accs.std()),
            cv_auc_mean=float(aucs.mean()),
            cv_auc_std=float(aucs.std()),
            holdout_acc=h_acc,
            holdout_auc=h_auc,
            top_v_features=top_v,
            top_jb_features=top_jb,
        )
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-name", default=DEFAULT_SEED,
                   help=f"Seed name to filter on (default: {DEFAULT_SEED})")
    p.add_argument("--transcripts", type=Path, default=TRANSCRIPT_DIR,
                   help=f"Transcript directory (default: {TRANSCRIPT_DIR})")
    args = p.parse_args()

    print(f"loading transcripts from {args.transcripts}/ "
          f"(seed='{args.seed_name}') ...")
    X, y, names = collect(seed=args.seed_name, transcript_dir=args.transcripts)
    n_v = int((y == "vanilla").sum())
    n_j = int((y == "jailbroken").sum())
    print(f"  n={len(y)}  (vanilla={n_v}, jailbroken={n_j})\n")

    print("=== descriptive stats by variant ===")
    describe_by_variant(X, y, names)

    print("\n=== univariate threshold classifiers (in-sample optimum) ===")
    print("    NOTE: these are upper bounds — chosen on the same data they're scored on.")
    univariate_threshold(X, y, names)

    print("\n=== logistic regression on all features ===")
    cv_logreg(X, y, names)

    print("\n=== single 70/30 held-out split ===")
    held_out_split(X, y)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
