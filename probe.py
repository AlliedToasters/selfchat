"""probes over per-message nomic embeddings (`emb_msgs.npz`).

Three probe targets, all on the same per-message embedding substrate:

  --target bucket
      Multi-class turn-depth. Bucket scheme via --scheme:
        thirds   = [0, 16, 33, 51)            # 3 classes (default)
        fives    = [0, 5, 10, ..., 50]        # 10 classes
        per-pair = [0, 2, 4, ..., 50]         # 25 classes
        per-turn = each turn_index its own class (≈50 classes)
      Or --bucket-edges "0,3,8,20,51" for a custom partition.

  --target turn-vs-all --max-j 10
      For each j in [0, max_j) trains a binary probe: "is this message from
      turn j?" vs "from anywhere else". Reports AUC / acc / MCC / base-rate
      per j. Cheap diagnostic for which individual turn indices are most
      separable from the rest of the conversation — surfaces a "decodability
      decay curve" without the cost of a 50-way softmax.

  --target agent
      Binary A vs B probe per variant. Within self-chat, A and B alternate
      strictly; this asks whether their outputs are content-distinguishable
      independent of turn parity.

  --target agent-by-depth --scheme fives
      Fits a per-bucket A-vs-B probe to surface where in the conversation
      role asymmetry concentrates. Each bucket has roughly equal A/B
      counts (parity-balanced for any contiguous range of ≥2 turns).
      Reports vanilla AUC vs jailbroken AUC per bucket.

All probes use L2-regularized softmax LR (binary = K=2 multinomial). Splits
are GroupKFold by run_id so a turn from a training-run never appears in
test. Two evaluations per target: within-variant CV (vanilla, jailbroken)
and cross-variant transfer (train on one, test on the other).

Run:
  python probe.py --target bucket --scheme thirds
  python probe.py --target turn-vs-all --max-j 10
  python probe.py --target agent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize


# --- bucket schemes --------------------------------------------------------

# Right edge 51 (not 50) so a literal turn_index of 50 (would only appear if
# n_turns somehow exceeded 50) doesn't fall outside the last bucket. With
# n_turns=50 the max valid index is 49, fully inside.
SCHEMES: dict[str, list[int]] = {
    "thirds": [0, 16, 33, 51],
    "fives": list(range(0, 51, 5)) + [51],
    "per-pair": list(range(0, 51, 2)) + [51],
    # per-turn is special-cased: y = turn_index directly.
}


def edges_to_buckets(edges: list[int]) -> list[tuple[int, int, str]]:
    """Convert sorted boundary list to (lo, hi, label) tuples."""
    out: list[tuple[int, int, str]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        out.append((lo, hi, f"{lo}-{hi - 1}"))
    return out


def bucket_y(turn_idx: np.ndarray, edges: list[int]) -> tuple[np.ndarray, list[str]]:
    """Map each turn_index to its bucket id; return (y, labels)."""
    buckets = edges_to_buckets(edges)
    y = np.full(len(turn_idx), -1, dtype=np.int64)
    for i, (lo, hi, _) in enumerate(buckets):
        y[(turn_idx >= lo) & (turn_idx < hi)] = i
    return y, [b[2] for b in buckets]


# --- multinomial LR via L-BFGS --------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _loss_grad(
    theta: np.ndarray, Xb: np.ndarray, Y: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    n, dp1 = Xb.shape
    k = Y.shape[1]
    W = theta.reshape(k, dp1)
    P = _softmax(Xb @ W.T)
    nll = -np.log((P * Y).sum(axis=1) + 1e-12).mean()
    reg = 0.5 * l2 * float((W[:, 1:] ** 2).sum())
    grad = (P - Y).T @ Xb / n
    grad[:, 1:] += l2 * W[:, 1:]
    return float(nll + reg), grad.ravel()


def fit_logreg(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X]).astype(np.float64)
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    theta0 = np.zeros(n_classes * (d + 1))
    res = optimize.minimize(
        _loss_grad,
        theta0,
        args=(Xb, Y, l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "disp": False},
    )
    return res.x.reshape(n_classes, d + 1)


def proba_logreg(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X]).astype(np.float64)
    return _softmax(Xb @ W.T)


# --- splits + metrics ------------------------------------------------------


def group_kfold(
    groups: np.ndarray, n_splits: int, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    fold_assign = {g: i % n_splits for i, g in enumerate(unique)}
    fold_of = np.array([fold_assign[g] for g in groups])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_splits)]


def cv_proba(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int, n_splits: int, l2: float
) -> np.ndarray:
    """Out-of-fold predicted probabilities (n, n_classes)."""
    proba = np.zeros((len(y), n_classes), dtype=np.float64)
    for train_idx, test_idx in group_kfold(groups, n_splits):
        W = fit_logreg(X[train_idx], y[train_idx], n_classes, l2=l2)
        proba[test_idx] = proba_logreg(X[test_idx], W)
    return proba


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def auc_binary(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann–Whitney U-based AUC; ties broken at 0.5. y in {0,1}."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average rank for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos_ranks = float(ranks[y == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mcc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0:
        return float("nan")
    return (tp * tn - fp * fn) / np.sqrt(denom)


# --- printing --------------------------------------------------------------


def print_confusion(cm: np.ndarray, labels: list[str], title: str) -> None:
    print()
    print(title)
    n = int(cm.sum())
    if n == 0:
        print("  (empty)")
        return
    acc = float(np.trace(cm)) / n
    # "always predict most common true class" baseline — what acc you'd get
    # by predicting the largest row-sum class regardless of input.
    majority = float(cm.sum(axis=1).max()) / n
    print(f"  acc = {acc:.3f}   majority-baseline = {majority:.3f}   n = {n}")

    if len(labels) <= 6:
        wide = max(6, max(len(l) for l in labels))
        print(f"  {'true ↓ / pred →':<22} " + "  ".join(f"{l:>{wide}}" for l in labels))
        for i, l in enumerate(labels):
            rs = int(cm[i].sum())
            counts = "  ".join(f"{cm[i, j]:>{wide}}" for j in range(len(labels)))
            pcts = (
                "  ".join(f"{cm[i, j] / rs:>5.1%}" if rs else "  n/a" for j in range(len(labels)))
                if rs
                else ""
            )
            print(f"  {l:<22} {counts}    {pcts}")
        return

    # Many classes: per-class recall summary + mean |true - pred| in
    # bucket-id units (treating bucket index as ordinal).
    diag = np.diag(cm)
    row_totals = cm.sum(axis=1)
    pred_idx = np.arange(len(labels))
    # mean abs error in class-index units, per row (true class) and overall
    overall_mae = 0.0
    overall_n = 0
    print(f"  {'true ↓':<14}  {'n':>6}  {'recall':>7}  {'mean|Δid|':>9}")
    for i, l in enumerate(labels):
        if row_totals[i] == 0:
            continue
        rec = diag[i] / row_totals[i]
        # per-row MAE in id units
        diffs = np.abs(pred_idx - i)
        mae = float((cm[i] * diffs).sum() / row_totals[i])
        overall_mae += float((cm[i] * diffs).sum())
        overall_n += int(row_totals[i])
        print(f"  {l:<14}  {row_totals[i]:>6}  {rec:>7.3f}  {mae:>9.2f}")
    if overall_n:
        print(f"  {'OVERALL':<14}  {overall_n:>6}  {acc:>7.3f}  {overall_mae / overall_n:>9.2f}")


# --- target dispatchers ----------------------------------------------------


def run_bucket(
    vecs: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    variants: np.ndarray,
    run_ids: np.ndarray,
    n_splits: int,
    l2: float,
) -> None:
    n_classes = len(labels)
    for variant in ("vanilla", "jailbroken"):
        mask = variants == variant
        n_groups = len(np.unique(run_ids[mask])) if mask.any() else 0
        if n_groups < n_splits:
            print(f"\n[{variant}] only {n_groups} runs — skipping CV.")
            continue
        proba = cv_proba(vecs[mask], y[mask], run_ids[mask], n_classes, n_splits, l2)
        pred = proba.argmax(axis=1)
        cm = confusion(y[mask], pred, n_classes)
        print_confusion(
            cm,
            labels,
            f"=== {variant}: {n_classes}-class bucket probe, "
            f"{n_splits}-fold CV by run_id (n_runs={n_groups}) ===",
        )

    for train_v, test_v in (("vanilla", "jailbroken"), ("jailbroken", "vanilla")):
        train_mask, test_mask = variants == train_v, variants == test_v
        if not (train_mask.any() and test_mask.any()):
            continue
        W = fit_logreg(vecs[train_mask], y[train_mask], n_classes, l2=l2)
        pred = proba_logreg(vecs[test_mask], W).argmax(axis=1)
        cm = confusion(y[test_mask], pred, n_classes)
        print_confusion(cm, labels, f"=== train={train_v}, test={test_v} (cross-variant) ===")


def run_turn_vs_all(
    vecs: np.ndarray,
    turn_idx: np.ndarray,
    variants: np.ndarray,
    run_ids: np.ndarray,
    n_splits: int,
    l2: float,
    max_j: int,
) -> None:
    for variant in ("vanilla", "jailbroken"):
        mask = variants == variant
        if not mask.any():
            continue
        n_groups = len(np.unique(run_ids[mask]))
        if n_groups < n_splits:
            print(f"\n[{variant}] only {n_groups} runs — skipping turn-vs-all.")
            continue
        Xv, tv, gv = vecs[mask], turn_idx[mask], run_ids[mask]
        print()
        print(
            f"=== {variant}: turn-j-vs-rest binary probes "
            f"({n_splits}-fold CV by run_id, n_runs={n_groups}) ==="
        )
        print(f"  {'j':>3}  {'n_pos':>6}  {'base':>6}  {'AUC':>6}  {'acc':>6}  {'MCC':>6}")
        for j in range(max_j):
            y_bin = (tv == j).astype(np.int64)
            n_pos = int(y_bin.sum())
            if n_pos < 2 or n_pos == len(y_bin):
                print(f"  {j:>3}  {n_pos:>6}  (insufficient positives)")
                continue
            base = n_pos / len(y_bin)
            proba = cv_proba(Xv, y_bin, gv, n_classes=2, n_splits=n_splits, l2=l2)[:, 1]
            auc = auc_binary(proba, y_bin)
            pred = (proba >= 0.5).astype(np.int64)
            acc = float((pred == y_bin).mean())
            mcc = mcc_binary(y_bin, pred)
            print(
                f"  {j:>3}  {n_pos:>6}  {base:>6.3f}  "
                f"{auc:>6.3f}  {acc:>6.3f}  {mcc:>6.3f}"
            )


def run_agent_by_depth(
    vecs: np.ndarray,
    agents: np.ndarray,
    turn_idx: np.ndarray,
    variants: np.ndarray,
    run_ids: np.ndarray,
    n_splits: int,
    l2: float,
    edges: list[int],
) -> None:
    """Per-bucket A-vs-B probe for vanilla and jailbroken side-by-side."""
    y_all = (agents == "B").astype(np.int64)
    buckets = edges_to_buckets(edges)

    print()
    print(
        f"=== agent A-vs-B AUC by turn-depth bucket "
        f"({n_splits}-fold CV by run_id) ==="
    )
    print(
        f"  {'bucket':<10}  {'V n':>5}  {'V AUC':>6}  "
        f"{'JB n':>5}  {'JB AUC':>6}  {'Δ (JB-V)':>8}"
    )
    for lo, hi, label in buckets:
        bmask = (turn_idx >= lo) & (turn_idx < hi)
        cells: dict[str, tuple[int, float]] = {}
        for variant in ("vanilla", "jailbroken"):
            mask = bmask & (variants == variant)
            n = int(mask.sum())
            n_groups = len(np.unique(run_ids[mask])) if mask.any() else 0
            n_pos = int(y_all[mask].sum())
            n_neg = n - n_pos
            if n_groups < n_splits or n_pos < 5 or n_neg < 5:
                cells[variant] = (n, float("nan"))
                continue
            proba = cv_proba(vecs[mask], y_all[mask], run_ids[mask], 2, n_splits, l2)[:, 1]
            auc = auc_binary(proba, y_all[mask])
            cells[variant] = (n, auc)
        v_n, v_auc = cells["vanilla"]
        jb_n, jb_auc = cells["jailbroken"]
        delta = jb_auc - v_auc if not (np.isnan(v_auc) or np.isnan(jb_auc)) else float("nan")
        v_str = f"{v_auc:>6.3f}" if not np.isnan(v_auc) else "   n/a"
        jb_str = f"{jb_auc:>6.3f}" if not np.isnan(jb_auc) else "   n/a"
        d_str = f"{delta:>+8.3f}" if not np.isnan(delta) else "    n/a"
        print(f"  {label:<10}  {v_n:>5}  {v_str}  {jb_n:>5}  {jb_str}  {d_str}")


def run_agent(
    vecs: np.ndarray,
    agents: np.ndarray,
    variants: np.ndarray,
    run_ids: np.ndarray,
    n_splits: int,
    l2: float,
) -> None:
    y = (agents == "B").astype(np.int64)
    for variant in ("vanilla", "jailbroken"):
        mask = variants == variant
        if not mask.any():
            continue
        n_groups = len(np.unique(run_ids[mask]))
        if n_groups < n_splits:
            print(f"\n[{variant}] only {n_groups} runs — skipping agent probe.")
            continue
        proba = cv_proba(vecs[mask], y[mask], run_ids[mask], 2, n_splits, l2)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        auc = auc_binary(proba, y[mask])
        acc = float((pred == y[mask]).mean())
        mcc = mcc_binary(y[mask], pred)
        n = int(mask.sum())
        print()
        print(
            f"=== {variant}: agent A-vs-B probe "
            f"({n_splits}-fold CV by run_id, n_runs={n_groups}, n_msgs={n}) ==="
        )
        print(f"  AUC = {auc:.3f}   acc = {acc:.3f}   MCC = {mcc:.3f}")

    for train_v, test_v in (("vanilla", "jailbroken"), ("jailbroken", "vanilla")):
        train_mask, test_mask = variants == train_v, variants == test_v
        if not (train_mask.any() and test_mask.any()):
            continue
        W = fit_logreg(vecs[train_mask], y[train_mask], 2, l2=l2)
        proba = proba_logreg(vecs[test_mask], W)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        auc = auc_binary(proba, y[test_mask])
        acc = float((pred == y[test_mask]).mean())
        mcc = mcc_binary(y[test_mask], pred)
        print(f"\n=== agent: train={train_v}, test={test_v} (cross-variant) ===")
        print(f"  AUC = {auc:.3f}   acc = {acc:.3f}   MCC = {mcc:.3f}")


# --- main ------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, nargs="?", default=Path("emb_msgs.npz"))
    p.add_argument(
        "--target",
        choices=("bucket", "turn-vs-all", "agent", "agent-by-depth"),
        default="bucket",
        help="Probe target axis",
    )
    p.add_argument("--scheme", default="thirds", help=f"Bucket scheme: one of {sorted(SCHEMES)} or 'per-turn'")
    p.add_argument(
        "--bucket-edges",
        default=None,
        help="Override --scheme with custom comma-separated edges (e.g. '0,3,8,20,51')",
    )
    p.add_argument("--max-j", type=int, default=10, help="For --target turn-vs-all: probe j=0..max_j-1")
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include degenerate-repetition runs (default: completed-only)",
    )
    p.add_argument(
        "--seed",
        default="all",
        help="Restrict to a specific seed condition (default: all)",
    )
    args = p.parse_args()

    if not args.npz.exists():
        print(f"not found: {args.npz}\n\nrun `python embed_messages.py --out {args.npz}` first.")
        return 2

    data = np.load(args.npz, allow_pickle=False)
    vecs = data["vectors"].astype(np.float32)
    turn_idx = data["turn_index"].astype(np.int64)
    variants = data["variants"].astype(str)
    seeds = data["seeds"].astype(str)
    run_ids = data["run_ids"].astype(str)
    stops = data["stop_reasons"].astype(str)
    agents = data["agents"].astype(str)

    print(f"loaded: n={len(vecs)} messages, dim={vecs.shape[1]}")
    print(f"  variants: {sorted(set(variants.tolist()))}")
    print(f"  seeds:    {sorted(set(seeds.tolist()))}")
    print(f"  stops:    {sorted(set(stops.tolist()))}")

    if not args.include_degenerate:
        keep = stops == "completed"
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"\nfiltering to completed-only: dropped {n_drop} (use --include-degenerate to keep)")
        vecs, turn_idx, variants, seeds, run_ids, stops, agents = (
            vecs[keep], turn_idx[keep], variants[keep], seeds[keep],
            run_ids[keep], stops[keep], agents[keep],
        )

    if args.seed != "all":
        keep = seeds == args.seed
        if not keep.any():
            print(f"seed {args.seed!r} not present after filtering.")
            return 1
        vecs, turn_idx, variants, seeds, run_ids, agents = (
            vecs[keep], turn_idx[keep], variants[keep], seeds[keep],
            run_ids[keep], agents[keep],
        )
        print(f"\nrestricted to seed={args.seed!r}: n={len(vecs)}")

    if args.target == "bucket":
        if args.bucket_edges is not None:
            edges = sorted({int(x) for x in args.bucket_edges.split(",")})
            if len(edges) < 2:
                print("--bucket-edges needs at least 2 values.")
                return 1
            y, labels = bucket_y(turn_idx, edges)
        elif args.scheme == "per-turn":
            # one bucket per integer turn_index actually present in the data
            present = sorted({int(i) for i in np.unique(turn_idx)})
            mapping = {t: i for i, t in enumerate(present)}
            y = np.array([mapping[int(t)] for t in turn_idx], dtype=np.int64)
            labels = [str(t) for t in present]
        elif args.scheme in SCHEMES:
            y, labels = bucket_y(turn_idx, SCHEMES[args.scheme])
        else:
            print(f"unknown --scheme {args.scheme!r}; choices: {sorted(SCHEMES)} + 'per-turn'")
            return 1

        print(f"\nbucket scheme: {args.scheme}, n_classes={len(labels)}")
        run_bucket(vecs, y, labels, variants, run_ids, args.n_splits, args.l2)

    elif args.target == "turn-vs-all":
        run_turn_vs_all(vecs, turn_idx, variants, run_ids, args.n_splits, args.l2, args.max_j)

    elif args.target == "agent":
        run_agent(vecs, agents, variants, run_ids, args.n_splits, args.l2)

    elif args.target == "agent-by-depth":
        if args.bucket_edges is not None:
            edges = sorted({int(x) for x in args.bucket_edges.split(",")})
        elif args.scheme in SCHEMES:
            edges = SCHEMES[args.scheme]
        else:
            print(f"--target agent-by-depth needs --scheme in {sorted(SCHEMES)} or --bucket-edges")
            return 1
        run_agent_by_depth(
            vecs, agents, turn_idx, variants, run_ids, args.n_splits, args.l2, edges
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
