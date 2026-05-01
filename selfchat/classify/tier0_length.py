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

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

TRANSCRIPT_DIR = Path("transcripts")

# Match `<variant>_freedom_<32hex>_<timestamp>.jsonl` only.
# Excludes freedom_dark, freedom_thirst, freedom_neg_minimal by requiring
# a 32-hex run_id immediately after `freedom_`.
FREEDOM_RE = re.compile(
    r"^(?P<variant>vanilla|jailbroken)_freedom_(?P<run_id>[0-9a-f]{32})_"
)

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


def extract_features(path: Path) -> tuple[str, dict[str, float]] | None:
    """Return (variant, features) for one transcript, or None if malformed."""
    m = FREEDOM_RE.match(path.name)
    if m is None:
        return None
    variant = m.group("variant")

    header = None
    footer = None
    contents: list[str] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ti = obj.get("turn_index")
            if ti == -1:
                header = obj
            elif ti == -2:
                footer = obj
            else:
                contents.append(obj.get("content", ""))

    if header is None or footer is None or not contents:
        return None

    lengths = np.array([len(c) for c in contents], dtype=np.float64)
    feats: dict[str, float] = {
        "completed_turns": float(footer.get("completed_turns", len(contents))),
        "is_early_stop": float(footer.get("completed_turns", 50) < 50),
        "stop_reason_degenerate": float(
            footer.get("stop_reason") == "degenerate_repetition"
        ),
        "total_chars": float(lengths.sum()),
        "mean_msg_chars": float(lengths.mean()),
        "median_msg_chars": float(np.median(lengths)),
        "std_msg_chars": float(lengths.std()),
        "max_msg_chars": float(lengths.max()),
        "min_msg_chars": float(lengths.min()),
    }
    return variant, feats


def collect() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[tuple[str, list[float]]] = []
    for p in sorted(TRANSCRIPT_DIR.iterdir()):
        result = extract_features(p)
        if result is None:
            continue
        variant, feats = result
        rows.append((variant, [feats[k] for k in FEATURES]))

    if not rows:
        raise RuntimeError(f"no freedom transcripts under {TRANSCRIPT_DIR}/")

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


def cv_logreg(X: np.ndarray, y: np.ndarray, names: list[str], seed: int = 0) -> None:
    y_bin = (y == "jailbroken").astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, aucs = [], []
    coefs: list[np.ndarray] = []
    for tr, te in skf.split(X, y_bin):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(
            Xtr, y_bin[tr]
        )
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y_bin[te], pred))
        aucs.append(roc_auc_score(y_bin[te], proba))
        coefs.append(clf.coef_.ravel())

    accs_a = np.array(accs)
    aucs_a = np.array(aucs)
    print(
        f"  5-fold CV: acc = {accs_a.mean():.3f} ± {accs_a.std():.3f}  "
        f"|  AUC = {aucs_a.mean():.3f} ± {aucs_a.std():.3f}"
    )

    # Coefficients trained on full data (averaged + final fit) for interpretation
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(
        scaler.transform(X), y_bin
    )
    print("\n  full-data coefficients (standardized features; +→ JB, −→ V):")
    order = np.argsort(-np.abs(clf.coef_.ravel()))
    for idx in order:
        print(f"    {names[idx]:<24}  {clf.coef_.ravel()[idx]:+.3f}")


def held_out_split(X: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
    """One 70/30 stratified split for a sanity check."""
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
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, pred)
    print(
        f"  70/30 holdout (seed={seed}): n_train={len(ytr)}, n_test={len(yte)}"
    )
    print(f"  acc = {acc:.3f}  |  AUC = {auc:.3f}")
    print("  confusion matrix [rows=true V/JB, cols=pred V/JB]:")
    print(f"    [[{cm[0,0]:>3} {cm[0,1]:>3}]")
    print(f"     [{cm[1,0]:>3} {cm[1,1]:>3}]]")


def main() -> int:
    print(f"loading transcripts from {TRANSCRIPT_DIR}/ ...")
    X, y, names = collect()
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
