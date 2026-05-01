"""Tier 1 null-hypothesis test: separate vanilla vs. jailbroken self-chat
rollouts on the `freedom` seed using TF-IDF features over message content.

One document per transcript = concatenation of all turn `content` strings.
Vocabulary is fit INSIDE each CV fold (sklearn Pipeline) to prevent
train/test leakage.

Two configs:
  * unigrams only        (1-grams, min_df=2)
  * unigrams + bigrams   (1- and 2-grams, min_df=2)

Run:
  python classify_freedom_tier1.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.model_selection import StratifiedKFold, train_test_split  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # type: ignore[import-not-found]
from sklearn.pipeline import Pipeline  # type: ignore[import-not-found]

from selfchat.core.shared import _STOPWORDS

TRANSCRIPT_DIR = Path("transcripts")
STOPWORDS = list(_STOPWORDS)

FREEDOM_RE = re.compile(
    r"^(?P<variant>vanilla|jailbroken)_freedom_(?P<run_id>[0-9a-f]{32})_"
)


def load_freedom_transcripts() -> tuple[list[str], np.ndarray, list[str]]:
    """Return (documents, variants, run_ids) — one document per transcript."""
    docs: list[str] = []
    variants: list[str] = []
    run_ids: list[str] = []
    for p in sorted(TRANSCRIPT_DIR.iterdir()):
        m = FREEDOM_RE.match(p.name)
        if m is None:
            continue
        contents: list[str] = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("turn_index", -1) >= 0:
                    contents.append(obj.get("content", ""))
        if not contents:
            continue
        docs.append("\n".join(contents))
        variants.append(m.group("variant"))
        run_ids.append(m.group("run_id"))
    if not docs:
        raise RuntimeError(f"no freedom transcripts under {TRANSCRIPT_DIR}/")
    return docs, np.array(variants), run_ids


def cv_eval(
    docs: list[str],
    y_bin: np.ndarray,
    ngram_range: tuple[int, int],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, aucs = [], []
    docs_a = np.array(docs, dtype=object)
    for tr, te in skf.split(docs_a, y_bin):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=2,
                lowercase=True,
                sublinear_tf=True,
                stop_words=STOPWORDS,
            )),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ])
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
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            lowercase=True,
            sublinear_tf=True,
            stop_words=STOPWORDS,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
    ])
    pipe.fit(Xtr.tolist(), ytr)
    proba = pipe.predict_proba(Xte.tolist())[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, pred)
    vocab = pipe.named_steps["tfidf"].get_feature_names_out().tolist()
    return float(acc), float(auc), cm, pipe, vocab


def report_top_features(pipe: Pipeline, vocab: list[str], k: int = 25) -> None:
    coef = pipe.named_steps["clf"].coef_.ravel()
    # +coef → JB, −coef → V
    order_jb = np.argsort(-coef)[:k]
    order_v = np.argsort(coef)[:k]
    print(f"\n  top {k} JB-leaning tokens (positive coef):")
    for idx in order_jb:
        print(f"    {coef[idx]:+.3f}  {vocab[idx]!r}")
    print(f"\n  top {k} V-leaning tokens (negative coef):")
    for idx in order_v:
        print(f"    {coef[idx]:+.3f}  {vocab[idx]!r}")


def run_config(
    docs: list[str],
    y_bin: np.ndarray,
    ngram_range: tuple[int, int],
    label: str,
) -> None:
    print(f"\n=== {label} (ngram_range={ngram_range}) ===")
    accs, aucs = cv_eval(docs, y_bin, ngram_range)
    print(f"  5-fold CV: acc = {accs.mean():.3f} ± {accs.std():.3f}  "
          f"|  AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")

    acc, auc, cm, pipe, vocab = holdout_eval(docs, y_bin, ngram_range)
    print(f"  70/30 holdout (seed=0):")
    print(f"    vocab size = {len(vocab)}")
    print(f"    acc = {acc:.3f}  |  AUC = {auc:.3f}")
    print("    confusion matrix [rows=true V/JB, cols=pred V/JB]:")
    print(f"      [[{cm[0,0]:>3} {cm[0,1]:>3}]")
    print(f"       [{cm[1,0]:>3} {cm[1,1]:>3}]]")
    report_top_features(pipe, vocab, k=20)


def main() -> int:
    print(f"loading transcripts from {TRANSCRIPT_DIR}/ ...")
    docs, variants, _ = load_freedom_transcripts()
    n_v = int((variants == "vanilla").sum())
    n_j = int((variants == "jailbroken").sum())
    avg_chars = np.mean([len(d) for d in docs])
    print(f"  n={len(docs)}  (vanilla={n_v}, jailbroken={n_j})")
    print(f"  mean transcript length: {avg_chars:.0f} chars")

    y_bin = (variants == "jailbroken").astype(int)

    run_config(docs, y_bin, (1, 1), "unigrams")
    run_config(docs, y_bin, (1, 2), "unigrams + bigrams")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
