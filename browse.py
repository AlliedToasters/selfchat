"""browser: interactive PCA + transcript viewer.

Loads embeddings produced by `embed.py`, projects to 2D PCA, renders an
interactive Plotly scatter where:

  * hovering a point shows top distinguishing tokens (TF-IDF over the same
    last-K terminal-state text that was embedded), so each point's hover
    surfaces what makes it lexically distinctive vs the rest of the corpus;
  * clicking a point loads the full pretty-formatted transcript in a side
    panel for inspection.

Use it to visually identify candidate clusters and label them by reading
their members. No clustering is computed automatically — that's Phase 2b.

Run: `streamlit run browse.py -- emb.npz`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from analyze import load
from embed import WORD_RE, terminal_state
from shared import _STOPWORDS

# Keep in sync with plot.py — same color encoding for visual continuity
# between the static figure and this browser.
HUE_FAMILIES = {"vanilla": "Blues", "jailbroken": "Reds"}
SEED_VISUAL_ORDER = ["task", "freedom", "alpaca", "advbench", "jbb_sans_advbench", "jbb"]
MARKER_MAP = {"completed": "circle", "degenerate_repetition": "x"}


@st.cache_data
def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "vectors": data["vectors"],
        "run_ids": data["run_ids"].astype(str),
        "variants": data["variants"].astype(str),
        "seeds": data["seeds"].astype(str),
        "stop_reasons": data["stop_reasons"].astype(str),
        "last_k": int(data["last_k"]) if "last_k" in data.files else 5,
        "model": str(data["model"]) if "model" in data.files else "unknown",
    }


@st.cache_data
def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2)), np.zeros(2)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    proj = xc @ vt[:2].T
    total = float((s**2).sum()) or 1.0
    evr = (s[:2] ** 2) / total
    return proj, evr


@st.cache_data
def index_transcripts(transcripts_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    d = Path(transcripts_dir)
    if not d.exists():
        return out
    for path in d.glob("*.jsonl"):
        try:
            with path.open() as f:
                header = json.loads(f.readline())
        except (OSError, json.JSONDecodeError):
            continue
        rid = header.get("run_id")
        if rid:
            out[rid] = str(path)
    return out


@st.cache_data
def load_terminal_texts(items: tuple[tuple[str, str], ...], last_k: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for rid, path in items:
        t = load(Path(path))
        out[rid] = terminal_state(t, last_k) if t is not None else ""
    return out

def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in WORD_RE.findall(text) if tok.lower() not in _STOPWORDS]


@st.cache_data
def compute_top_tokens(texts: tuple[str, ...], top_n: int) -> list[list[tuple[str, float]]]:
    """Per-doc TF-IDF top tokens (sklearn-smooth IDF, L2-normalized rows).

    TF: count(token) / total tokens in doc.
    IDF: log((N + 1) / (df + 1)) + 1.
    Row L2-normalized so scores are comparable across docs of different length.
    """
    n = len(texts)
    if n == 0:
        return []
    docs = [tokenize(t) for t in texts]

    df: dict[str, int] = {}
    for tokens in docs:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: float(np.log((n + 1) / (cnt + 1)) + 1.0) for tok, cnt in df.items()}

    out: list[list[tuple[str, float]]] = []
    for tokens in docs:
        if not tokens:
            out.append([])
            continue
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        total = len(tokens)
        raw = np.array([(cnt / total) * idf[tok] for tok, cnt in tf.items()])
        norm = float(np.linalg.norm(raw)) or 1.0
        raw = raw / norm
        scored = sorted(zip(tf.keys(), raw, strict=True), key=lambda kv: kv[1], reverse=True)
        out.append([(tok, float(s)) for tok, s in scored[:top_n]])
    return out


def build_palette(variants_in_data: list[str], seeds_in_data: list[str]) -> dict[str, str]:
    seeds_ordered = [s for s in SEED_VISUAL_ORDER if s in set(seeds_in_data)] + sorted(
        set(seeds_in_data) - set(SEED_VISUAL_ORDER)
    )
    palette: dict[str, str] = {}
    for v in sorted(variants_in_data):
        family = HUE_FAMILIES.get(v, "Greys")
        shades = sns.color_palette(family, n_colors=len(seeds_ordered) + 2)[1:-1]
        for i, s in enumerate(seeds_ordered):
            r, g, b = shades[i]
            palette[f"{v}|{s}"] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return palette


def render_transcript(path: Path) -> None:
    if not path.exists():
        st.warning(f"transcript not found: {path}")
        return
    lines = path.read_text().splitlines()
    if not lines:
        st.info("(empty transcript)")
        return

    header = json.loads(lines[0])
    st.markdown(
        f"### {header['model_variant']} / `{header['seed_name']}` · run `{header['run_id'][:8]}`"
    )
    st.caption(
        f"model_tag: `{header['model_tag']}` · n_turns target: {header['n_turns']} · "
        f"started: {header['started_at']}"
    )
    with st.expander("seed prompt", expanded=False):
        st.markdown(f"_{header['seed_prompt']}_")

    st.markdown("---")

    footer: dict | None = None
    for line in lines[1:]:
        rec = json.loads(line)
        if rec.get("turn_index") == -2:
            footer = rec
            continue
        agent = rec["agent"]
        with st.chat_message("user" if agent == "A" else "assistant"):
            st.caption(f"turn {rec['turn_index']:02d} · agent {agent} · {rec['elapsed_ms']} ms")
            st.markdown(rec["content"])

    st.markdown("---")
    if footer:
        st.markdown(
            f"**stop_reason:** `{footer['stop_reason']}` · "
            f"completed_turns: {footer['completed_turns']} / {header['n_turns']} · "
            f"finished: {footer['finished_at']}"
        )
    else:
        st.warning("no footer — older transcript or run did not finish cleanly")


def main() -> None:
    st.set_page_config(page_title="hidden attractors browser", layout="wide")
    st.title("hidden attractors — terminal-state browser")

    extra = sys.argv[1:]
    npz_path = Path(extra[0]) if extra else Path("emb.npz")
    transcript_dir = Path(extra[1]) if len(extra) > 1 else Path("transcripts")
    if not npz_path.exists():
        st.error(f"not found: {npz_path}\n\nrun `python embed.py --out {npz_path}` first.")
        return

    raw = load_npz(str(npz_path))
    proj, evr = pca_2d(raw["vectors"])

    path_index = index_transcripts(str(transcript_dir))
    items = tuple((rid, path_index[rid]) for rid in raw["run_ids"] if rid in path_index)
    texts_map = load_terminal_texts(items, raw["last_k"])
    texts_in_order = tuple(texts_map.get(rid, "") for rid in raw["run_ids"])

    with st.sidebar:
        st.header("filters")
        all_variants = sorted(set(raw["variants"].tolist()))
        all_seeds = sorted(set(raw["seeds"].tolist()))
        all_stops = sorted(set(raw["stop_reasons"].tolist()))
        sel_variants = st.multiselect("variants", all_variants, default=all_variants)
        sel_seeds = st.multiselect("seeds", all_seeds, default=all_seeds)
        sel_stops = st.multiselect("stop reasons", all_stops, default=all_stops)
        top_n = st.slider("top tokens per point", min_value=3, max_value=15, value=8)
        st.markdown("---")
        st.markdown(
            f"**dataset**\n\n"
            f"- model: `{raw['model']}`\n"
            f"- last_k: {raw['last_k']}\n"
            f"- n_total: {len(raw['run_ids'])}\n"
            f"- PC1: {evr[0]:.1%}\n"
            f"- PC2: {evr[1]:.1%}"
        )
        missing = len(raw["run_ids"]) - len(items)
        if missing:
            st.caption(f"({missing} run_id(s) had no matching transcript file)")

    top_tokens = compute_top_tokens(texts_in_order, top_n=top_n)
    cell = np.array([f"{v}|{s}" for v, s in zip(raw["variants"], raw["seeds"], strict=True)])
    df = pd.DataFrame(
        {
            "PC1": proj[:, 0],
            "PC2": proj[:, 1],
            "run_id": raw["run_ids"],
            "variant": raw["variants"],
            "seed": raw["seeds"],
            "stop": raw["stop_reasons"],
            "cell": cell,
            "top_tokens": [
                "<br>".join(f"&nbsp;&nbsp;{t}&nbsp;&nbsp;<i>({s:.2f})</i>" for t, s in toks)
                if toks
                else "<i>(no tokens)</i>"
                for toks in top_tokens
            ],
        }
    )

    mask = (
        df["variant"].isin(sel_variants) & df["seed"].isin(sel_seeds) & df["stop"].isin(sel_stops)
    )
    df_show = df[mask].reset_index(drop=True)

    if df_show.empty:
        st.warning("filter excluded every point.")
        return

    palette = build_palette(
        list(df_show["variant"].unique()),
        list(df_show["seed"].unique()),
    )
    cells_in_view = sorted(df_show["cell"].unique())
    stops_in_view = sorted(df_show["stop"].unique())

    fig = px.scatter(
        df_show,
        x="PC1",
        y="PC2",
        color="cell",
        symbol="stop",
        category_orders={"cell": cells_in_view, "stop": stops_in_view},
        symbol_map=MARKER_MAP,
        color_discrete_map=palette,
        custom_data=["run_id", "variant", "seed", "stop", "top_tokens"],
        height=620,
    )
    fig.update_traces(
        marker=dict(size=11, line=dict(width=0.6, color="black")),
        hovertemplate=(
            "<b>%{customdata[1]} | %{customdata[2]}</b><br>"
            "stop: %{customdata[3]}<br>"
            "run: %{customdata[0]}<br>"
            "<br><b>top tokens (tf-idf):</b><br>%{customdata[4]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        legend_title_text="",
        xaxis_title=f"PC1 ({evr[0]:.1%})",
        yaxis_title=f"PC2 ({evr[1]:.1%})",
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
    )

    col_plot, col_panel = st.columns([3, 2], gap="large")

    with col_plot:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode=("points",),
            key="pca_scatter",
        )

    with col_panel:
        st.subheader("transcript")
        selected = []
        if event and isinstance(event, dict):
            selected = event.get("selection", {}).get("points", []) or []
        if not selected:
            st.info("Click a point in the scatter to load its transcript here.")
        else:
            point = selected[0]
            cd = point.get("customdata") or []
            run_id = cd[0] if cd else None
            if not run_id:
                st.warning("could not resolve run_id from selection")
            else:
                path = path_index.get(run_id)
                if not path:
                    st.warning(f"no transcript file found for run_id {run_id[:8]}")
                else:
                    container = st.container(height=720)
                    with container:
                        render_transcript(Path(path))


if __name__ == "__main__":
    main()
