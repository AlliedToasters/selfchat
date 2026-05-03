"""Tier 1 null-hypothesis test: separate vanilla vs. jailbroken self-chat
rollouts on the `freedom` seed using a bag-of-characters representation.

Sits between Tier 0 (length+completion only — no content) and Tier 2
(word-level TF-IDF). Character frequencies are roughly invariant to
vocabulary choice but pick up punctuation, casing, formatting, and
character-class drift (e.g. heavy use of digits, em-dashes, asterisks,
stars). If V vs JB is separable on this representation, the difference
is partly *typographic*, not just lexical.

One document per transcript = concatenation of all turn `content` strings.
TfidfVectorizer is fit INSIDE each CV fold (sklearn Pipeline) to prevent
train/test leakage.

Two configs:
  * char unigrams       (analyzer='char', ngram_range=(1,1))   — true bag-of-characters
  * char unigrams+bigrams (analyzer='char', ngram_range=(1,2)) — adjacent-pair extension

Run:
  python -m selfchat.classify.tier1_chars
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # type: ignore[import-not-found]
from sklearn.pipeline import Pipeline  # type: ignore[import-not-found]

from selfchat.analysis.analyze import NATURAL_STOPS, load as load_transcript
from selfchat.classify._common import DEFAULT_SEED, TierResult, make_seed_re

TRANSCRIPT_DIR = Path("transcripts")


def load_transcripts(
    seed: str = DEFAULT_SEED, transcript_dir: Path = TRANSCRIPT_DIR
) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (documents, variants, run_ids) — one document per transcript.

    Uses `analyze.load()` for robust JSONL parsing (in-flight files with
    truncated last lines are returned as `stop_reason='in_flight'` and
    filtered out by the NATURAL_STOPS check).
    """
    seed_re = make_seed_re(seed)
    docs: list[str] = []
    variants: list[str] = []
    run_ids: list[str] = []
    for p in sorted(transcript_dir.iterdir()):
        m = seed_re.match(p.name)
        if m is None:
            continue
        t = load_transcript(p)
        if t is None or t.stop_reason not in NATURAL_STOPS or not t.turns:
            continue
        contents = [turn.content for turn in t.turns]
        if not contents:
            continue
        docs.append("\n".join(contents))
        variants.append(m.group("variant"))
        run_ids.append(m.group("run_id"))
    if not docs:
        raise RuntimeError(
            f"no transcripts matching seed='{seed}' under {transcript_dir}/"
        )
    return docs, np.array(variants), run_ids


def make_pipeline(ngram_range: tuple[int, int], seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=2,
            lowercase=False,            # keep case — capitalization patterns may differ
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
    ])


def cv_eval(
    docs: list[str],
    y_bin: np.ndarray,
    ngram_range: tuple[int, int],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    accs, aucs = [], []
    docs_a = np.array(docs, dtype=object)
    for tr, te in skf.split(docs_a, y_bin):
        pipe = make_pipeline(ngram_range, seed)
        pipe.fit(docs_a[tr].tolist(), y_bin[tr])
        proba = pipe.predict_proba(docs_a[te].tolist())[:, 1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y_bin[te], pred))
        aucs.append(roc_auc_score(y_bin[te], proba))
    return np.array(accs), np.array(aucs)


def holdout_eval(
    docs: list[str],
    y_bin: np.ndarray,
    ngram_range: tuple[int, int],
    seed: int = 0,
) -> tuple[float, float, np.ndarray, Pipeline, list[str]]:
    docs_a = np.array(docs, dtype=object)
    Xtr, Xte, ytr, yte = train_test_split(
        docs_a, y_bin, test_size=0.3, stratify=y_bin, random_state=seed
    )
    pipe = make_pipeline(ngram_range, seed)
    pipe.fit(list(Xtr), ytr)
    proba = pipe.predict_proba(list(Xte))[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, pred)
    vocab = pipe.named_steps["tfidf"].get_feature_names_out().tolist()
    return float(acc), float(auc), cm, pipe, vocab


def _display(tok: str) -> str:
    """Render a char or char-bigram so whitespace/control chars are visible."""
    return repr(tok)


def report_top_features(pipe: Pipeline, vocab: list[str], k: int = 25) -> None:
    coef = pipe.named_steps["clf"].coef_.ravel()
    # +coef → JB, −coef → V
    order_jb = np.argsort(-coef)[:k]
    order_v = np.argsort(coef)[:k]
    print(f"\n  top {k} JB-leaning chars/n-grams (positive coef):")
    for idx in order_jb:
        print(f"    {coef[idx]:+.3f}  {_display(vocab[idx])}")
    print(f"\n  top {k} V-leaning chars/n-grams (negative coef):")
    for idx in order_v:
        print(f"    {coef[idx]:+.3f}  {_display(vocab[idx])}")


def run_config(
    docs: list[str],
    y_bin: np.ndarray,
    ngram_range: tuple[int, int],
    label: str,
) -> None:
    print(f"\n=== {label} (analyzer='char', ngram_range={ngram_range}) ===")
    accs, aucs = cv_eval(docs, y_bin, ngram_range)
    print(f"  5×5 repeated CV: acc = {accs.mean():.3f} ± {accs.std():.3f}  "
          f"|  AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")

    acc, auc, cm, pipe, vocab = holdout_eval(docs, y_bin, ngram_range)
    print(f"  70/30 holdout (seed=0):")
    print(f"    vocab size = {len(vocab)}")
    print(f"    acc = {acc:.3f}  |  AUC = {auc:.3f}")
    print("    confusion matrix [rows=true V/JB, cols=pred V/JB]:")
    print(f"      [[{cm[0,0]:>3} {cm[0,1]:>3}]")
    print(f"       [{cm[1,0]:>3} {cm[1,1]:>3}]]")
    report_top_features(pipe, vocab, k=20)


def _full_data_top_tokens(
    docs: list[str], y_bin: np.ndarray, ngram_range: tuple[int, int], k: int
) -> tuple[list[str], list[str]]:
    """Fit on all data and return (top-V tokens, top-JB tokens)."""
    pipe = make_pipeline(ngram_range, seed=0)
    pipe.fit(docs, y_bin)
    coef = pipe.named_steps["clf"].coef_.ravel()
    vocab = pipe.named_steps["tfidf"].get_feature_names_out().tolist()
    order_jb = np.argsort(-coef)[:k]
    order_v = np.argsort(coef)[:k]
    return ([vocab[i] for i in order_v], [vocab[i] for i in order_jb])


def run(
    seed: str = DEFAULT_SEED,
    transcript_dir: Path = TRANSCRIPT_DIR,
    k_features: int = 3,
) -> list[TierResult]:
    docs, variants, _ = load_transcripts(seed, transcript_dir)
    y_bin = (variants == "jailbroken").astype(int)
    n = len(docs)

    out: list[TierResult] = []
    for ngram_range, label in [((1, 1), "char (1,1)"), ((1, 2), "char (1,2)")]:
        accs, aucs = cv_eval(docs, y_bin, ngram_range)
        h_acc, h_auc, _, _, _ = holdout_eval(docs, y_bin, ngram_range)
        v_top, jb_top = _full_data_top_tokens(docs, y_bin, ngram_range, k_features)
        out.append(
            TierResult(
                tier="1",
                mode=label,
                n=n,
                cv_acc_mean=float(accs.mean()),
                cv_acc_std=float(accs.std()),
                cv_auc_mean=float(aucs.mean()),
                cv_auc_std=float(aucs.std()),
                holdout_acc=h_acc,
                holdout_auc=h_auc,
                top_v_features=[repr(t) for t in v_top],
                top_jb_features=[repr(t) for t in jb_top],
            )
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-name", default=DEFAULT_SEED,
                   help=f"Seed name to filter on (default: {DEFAULT_SEED})")
    p.add_argument("--transcripts", type=Path, default=TRANSCRIPT_DIR,
                   help=f"Transcript directory (default: {TRANSCRIPT_DIR})")
    args = p.parse_args()

    print(f"loading transcripts from {args.transcripts}/ "
          f"(seed='{args.seed_name}') ...")
    docs, variants, _ = load_transcripts(args.seed_name, args.transcripts)
    n_v = int((variants == "vanilla").sum())
    n_j = int((variants == "jailbroken").sum())
    avg_chars = np.mean([len(d) for d in docs])
    print(f"  n={len(docs)}  (vanilla={n_v}, jailbroken={n_j})")
    print(f"  mean transcript length: {avg_chars:.0f} chars")

    y_bin = (variants == "jailbroken").astype(int)

    run_config(docs, y_bin, (1, 1), "char unigrams (true bag-of-characters)")
    run_config(docs, y_bin, (1, 2), "char unigrams + bigrams")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
