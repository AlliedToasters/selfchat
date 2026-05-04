"""Cluster Lab — interactive per-message clustering of nomic embeddings.

Recipe (matches `selfchat.stats.jb_purity_balanced`):

  1. Load `artifacts/emb_msgs.npz`.
  2. Filter completed-only and len ≥ min_chars (default 150).
  3. Balance per (seed, variant) cell to target_n (default = smallest
     cell). This is the *fit* substrate.
  4. Fit KMeans(k) on the balanced fit substrate.
  5. Predict labels on the *full* (pre-balance) filtered substrate by
     nearest centroid → these are the labels we visualise.
  6. Project to 2D via PCA or t-SNE for the scatter.

Click a point to load its transcript on the right; the clicked turn is
anchored at the top of the scrollable pane.

Run:
  scripts/run_msg_analysis.sh
or directly:
  streamlit run selfchat/viz/browse.py
and use the sidebar to navigate to "1 cluster lab".
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from selfchat.stats.cluster import load_lengths_aligned
from selfchat.stats.purity_profile import balance_per_seed_variant


# --- caches ---------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "vectors": data["vectors"].astype(np.float32),
        "turn_index": data["turn_index"].astype(np.int64),
        "variants": data["variants"].astype(str),
        "seeds": data["seeds"].astype(str),
        "run_ids": data["run_ids"].astype(str),
        "stop_reasons": data["stop_reasons"].astype(str),
        "agents": data["agents"].astype(str),
    }


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner="loading message lengths from transcripts...")
def cached_lengths(
    transcript_dir: str, run_ids: tuple[str, ...], turn_idx: tuple[int, ...]
) -> np.ndarray:
    return load_lengths_aligned(
        Path(transcript_dir), np.asarray(run_ids), np.asarray(turn_idx)
    )


@st.cache_data(show_spinner="loading message texts...")
def cached_texts(
    transcript_dir: str, run_ids: tuple[str, ...], turn_idx: tuple[int, ...]
) -> tuple[str, ...]:
    """Per-row message text (or empty). Used for hover preview + transcript pane."""
    by_run: dict[str, dict[int, str]] = {}
    needed = set(run_ids)
    d = Path(transcript_dir)
    for path in sorted(d.glob("*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            continue
        rid = header.get("run_id")
        if not rid or rid not in needed:
            continue
        run_map: dict[int, str] = {}
        for line in lines[1:]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ti = rec.get("turn_index")
            if ti is None or ti < 0:
                continue
            run_map[int(ti)] = rec.get("content", "")
        by_run[rid] = run_map
    return tuple(
        by_run.get(rid, {}).get(int(ti), "") for rid, ti in zip(run_ids, turn_idx)
    )


@st.cache_data(show_spinner="loading llama-guard vet results...")
def load_vet_results(
    vet_path: str, transcript_dir: str
) -> dict:
    """Join vet_results.jsonl onto (run_id, turn_index) for color-by.

    Vet entries use the transcript basename as their key; we invert the
    transcript header lookup to recover run_id per file.
    """
    out = {
        "per_msg_p": {},  # (run_id, turn_idx) -> p_unsafe
        "per_msg_v": {},  # (run_id, turn_idx) -> "safe"|"unsafe"
        "per_run": {},    # run_id -> {max_p_unsafe, mean_p_unsafe, verdict}
    }
    if not Path(vet_path).exists():
        return out
    fname_to_runid: dict[str, str] = {}
    d = Path(transcript_dir)
    if d.exists():
        for path in d.glob("*.jsonl"):
            try:
                with path.open() as f:
                    header = json.loads(f.readline())
            except (OSError, json.JSONDecodeError):
                continue
            rid = header.get("run_id")
            if rid:
                fname_to_runid[path.name] = rid
    with open(vet_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = fname_to_runid.get(rec.get("file", ""))
            if not rid:
                continue
            out["per_run"][rid] = {
                "max_p_unsafe": float(rec.get("max_p_unsafe") or 0.0),
                "mean_p_unsafe": float(rec.get("mean_p_unsafe") or 0.0),
                "verdict": rec.get("verdict", "unknown"),
            }
            for t in rec.get("turns", []):
                ti = t.get("turn_index")
                if ti is None:
                    continue
                key = (rid, int(ti))
                out["per_msg_p"][key] = float(t.get("p_unsafe") or 0.0)
                out["per_msg_v"][key] = t.get("verdict", "unknown")
    return out


@st.cache_data(show_spinner="fitting kmeans (balanced) + assigning full substrate...")
def fit_balanced_assign_full(
    vectors: np.ndarray,
    seeds_arr: np.ndarray,
    variants: np.ndarray,
    target_n: int | None,
    balance: bool,
    k: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance → fit KMeans → predict on full.

    Returns (labels_full, balanced_idx, centroids).
    """
    from sklearn.cluster import KMeans  # type: ignore[import-not-found]

    if balance:
        balanced_idx = balance_per_seed_variant(
            seeds_arr, variants, target_n, rng_seed
        )
        fit_X = vectors[balanced_idx]
    else:
        balanced_idx = np.arange(len(vectors), dtype=np.int64)
        fit_X = vectors

    km = KMeans(
        n_clusters=k, random_state=rng_seed, n_init=10  # type: ignore[arg-type]
    )
    km.fit(fit_X)
    labels = np.asarray(km.predict(vectors), dtype=np.int64)
    centroids = np.asarray(km.cluster_centers_, dtype=np.float32)
    return labels, balanced_idx, centroids


@st.cache_data(show_spinner="computing PCA...")
def pca_2d(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (proj, evr, mean, components_2x_dim).

    `mean` and `components` let us project arbitrary held-out points
    (e.g. turns excluded by min_chars) into the same subspace.
    """
    if x.shape[0] < 2 or x.shape[1] < 2:
        return (
            np.zeros((x.shape[0], 2), dtype=np.float32),
            np.zeros(2),
            np.zeros(x.shape[1] if x.ndim == 2 else 0, dtype=np.float32),
            np.zeros((2, x.shape[1] if x.ndim == 2 else 0), dtype=np.float32),
        )
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:2].astype(np.float32)
    proj = (xc @ components.T).astype(np.float32)
    total = float((s**2).sum()) or 1.0
    evr = (s[:2] ** 2) / total
    return proj, evr, mean.astype(np.float32), components


@st.cache_data(show_spinner="computing t-SNE (this is slow on first run)...")
def tsne_2d(
    x: np.ndarray, perplexity: float, rng_seed: int
) -> np.ndarray:
    from sklearn.manifold import TSNE  # type: ignore[import-not-found]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=rng_seed,
        verbose=0,
    )
    return tsne.fit_transform(x).astype(np.float32)


# --- characterization -----------------------------------------------------


def cluster_summary_df(
    labels: np.ndarray,
    variants: np.ndarray,
    seeds_arr: np.ndarray,
    turn_idx: np.ndarray,
) -> pd.DataFrame:
    n_total = len(labels)
    base_jb = float((variants == "jailbroken").mean()) if n_total else 0.0
    rows: list[dict] = []
    for c in sorted({int(c) for c in np.unique(labels)}):
        mask = labels == c
        n = int(mask.sum())
        jb = int((variants[mask] == "jailbroken").sum())
        jb_frac = jb / n if n else 0.0
        enrich = jb_frac / base_jb if base_jb > 0 else float("nan")
        seed_counts = Counter(seeds_arr[mask].tolist())
        top_seed = seed_counts.most_common(1)[0] if seed_counts else ("", 0)
        t_in = turn_idx[mask] if n else np.array([0])
        rows.append(
            {
                "cid": c,
                "n": n,
                "frac": n / n_total if n_total else 0.0,
                "JB": jb,
                "V": n - jb,
                "JB%": jb_frac,
                "enrich": enrich,
                "turn_μ": float(t_in.mean()) if n else 0.0,
                "turn_med": float(np.median(t_in)) if n else 0.0,
                "top_seed": f"{top_seed[0]} ({top_seed[1] / n:.0%})" if n else "",
            }
        )
    return pd.DataFrame(rows)


# --- trajectory ----------------------------------------------------------


def run_trajectory(
    raw: dict[str, np.ndarray],
    run_id: str,
    centroids: np.ndarray,
) -> dict[str, np.ndarray]:
    """For all turns of `run_id` in the *raw* (unfiltered) npz, return:

      turn_index, vectors, assigned_cluster (nearest centroid),
      sims (cosine sim to each centroid, shape (n_turns, k)),
      stop_reasons, lengths-from-text-not-available-here so caller may add.

    Indexing is in the npz row order; turns may not be contiguous if
    transcripts had gaps, but kept as-recorded.
    """
    rids = raw["run_ids"]
    mask = rids == run_id
    idx = np.where(mask)[0]
    order = np.argsort(raw["turn_index"][idx])
    idx = idx[order]
    if len(idx) == 0:
        return {
            "turn_index": np.array([], dtype=np.int64),
            "assigned": np.array([], dtype=np.int64),
            "sims": np.zeros((0, centroids.shape[0]), dtype=np.float32),
            "stop_reasons": np.array([], dtype=str),
            "agents": np.array([], dtype=str),
        }
    V = raw["vectors"][idx]  # already L2-normalised in our pipeline
    # Cosine sim to each centroid: V @ c / (||V|| * ||c||); ||V|| = 1.
    cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cnorms = np.where(cnorms == 0, 1.0, cnorms)
    centroids_unit = centroids / cnorms
    sims = V @ centroids_unit.T  # (n_turns, k)
    assigned = sims.argmax(axis=1).astype(np.int64)
    return {
        "turn_index": raw["turn_index"][idx].astype(np.int64),
        "assigned": assigned,
        "sims": sims.astype(np.float32),
        "stop_reasons": raw["stop_reasons"][idx].astype(str),
        "agents": raw["agents"][idx].astype(str),
    }


def trajectory_figure(
    traj: dict[str, np.ndarray],
    focus_cluster: int,
    focus_turn: int,
    cluster_palette: dict[int, str],
):
    """Plotly figure: similarity-to-focus-centroid by turn, markers colored by
    each turn's assigned cluster; clicked turn marked with a vertical line."""
    import plotly.graph_objects as go

    turns = traj["turn_index"]
    sims_focus = traj["sims"][:, focus_cluster] if traj["sims"].size else np.array([])
    assigned = traj["assigned"]

    fig = go.Figure()
    # connecting line (no color encoding) so the trajectory is visible
    fig.add_trace(
        go.Scatter(
            x=turns,
            y=sims_focus,
            mode="lines",
            line=dict(width=1.5, color="rgba(120,120,120,0.6)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # one marker trace per cluster present (so legend doubles as cluster key)
    for c in sorted(set(assigned.tolist())):
        m = assigned == c
        color = cluster_palette.get(int(c), "#999999")
        size = np.where(turns[m] == focus_turn, 14, 9)
        fig.add_trace(
            go.Scatter(
                x=turns[m],
                y=sims_focus[m],
                mode="markers",
                name=f"cid={c}{' ★' if c == focus_cluster else ''}",
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=0.6, color="black"),
                    symbol="circle",
                ),
                hovertemplate=(
                    f"turn %{{x}}<br>sim={'%{y:.3f}'}<br>cid={c}<extra></extra>"
                ),
            )
        )
    fig.add_vline(
        x=focus_turn, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.55)"
    )
    fig.add_hline(
        y=0, line_width=0.5, line_dash="dot", line_color="rgba(0,0,0,0.25)"
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="turn_index",
        yaxis_title=f"cos-sim to centroid (cid={focus_cluster})",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(size=10),
        ),
    )
    return fig


def cluster_color_map(cluster_ids: list[int]) -> dict[int, str]:
    """Stable color per cluster id, matching plotly's default qualitative palette."""
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
    return {c: palette[i % len(palette)] for i, c in enumerate(cluster_ids)}


def _is_dark_theme() -> bool:
    """True iff streamlit's runtime theme reports dark mode.

    Uses st.context.theme.type (streamlit ≥ 1.46). Defaults to False if
    the field isn't populated (e.g. AppTest or older runtime).
    """
    try:
        ttype = st.context.theme.get("type")
    except Exception:
        return False
    return str(ttype).lower() == "dark"


def _add_path_overlay(
    fig,
    *,
    raw: dict[str, np.ndarray],
    run_id: str,
    focus_turn: int,
    focus_cluster: int,
    proj_method: str,
    run_ids_plot: np.ndarray,
    turn_idx_plot: np.ndarray,
    proj_plot: np.ndarray,
    plot_idx: np.ndarray,
    centroids: np.ndarray,
    cluster_palette: dict[int, str],
    pca_mean: np.ndarray | None,
    pca_components: np.ndarray | None,
    dark: bool = False,
) -> None:
    """Overlay the clicked run's conversation path as a connected polyline.

    For PCA we re-project ALL turns of the run (including ones excluded by
    min_chars) using the stored mean+components, so the path is complete.
    For t-SNE we only have projected coordinates for the subsample, so the
    path is restricted to whatever turns happened to land in the subsample.
    """
    import plotly.graph_objects as go

    if proj_method == "PCA" and pca_mean is not None and pca_components is not None:
        full_run_mask = raw["run_ids"] == run_id
        full_idx = np.where(full_run_mask)[0]
        if full_idx.size == 0:
            return
        order = np.argsort(raw["turn_index"][full_idx])
        full_idx = full_idx[order]
        V = raw["vectors"][full_idx]
        path_xy = ((V - pca_mean) @ pca_components.T).astype(np.float32)
        path_turns = raw["turn_index"][full_idx].astype(np.int64)
        # assign cluster per turn via nearest centroid (cosine on unit-norm V)
        cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
        cnorms = np.where(cnorms == 0, 1.0, cnorms)
        path_clusters = (V @ (centroids / cnorms).T).argmax(axis=1).astype(np.int64)
    else:
        # t-SNE — restrict to turns of this run that are in the plotted subsample.
        plot_set = set(plot_idx.tolist())
        run_mask = run_ids_plot == run_id
        run_pos = np.where(run_mask)[0]
        run_pos = np.array(
            [i for i in run_pos.tolist() if i in plot_set], dtype=np.int64
        )
        if run_pos.size == 0:
            return
        order = np.argsort(turn_idx_plot[run_pos])
        run_pos = run_pos[order]
        path_xy = proj_plot[run_pos]
        path_turns = turn_idx_plot[run_pos].astype(np.int64)
        # cluster per turn from cached labels would be best — caller passes
        # plot points only; fall back to nearest centroid on the embedding.
        # We don't have raw vectors here, so just nearest centroid is fine
        # since cluster predictions are nearest-centroid by construction.
        # Compute against unit-norm centroids using the embedding rows.
        # Fetch the embeddings for the in-plot run rows from `raw`.
        # (We can map back via run_id + turn_index.)
        rids_full = raw["run_ids"]
        tidx_full = raw["turn_index"]
        full_run_mask = rids_full == run_id
        full_idx_all = np.where(full_run_mask)[0]
        ti_to_full_idx = {int(tidx_full[i]): int(i) for i in full_idx_all.tolist()}
        path_clusters = np.array(
            [
                int(
                    (
                        raw["vectors"][ti_to_full_idx[int(t)]]
                        @ (centroids / np.linalg.norm(
                            centroids, axis=1, keepdims=True
                        ).clip(min=1e-12)).T
                    ).argmax()
                )
                if int(t) in ti_to_full_idx
                else -1
                for t in path_turns
            ],
            dtype=np.int64,
        )

    line_color = (
        "rgba(245,245,245,0.92)" if dark else "rgba(20,20,20,0.85)"
    )
    text_color = "rgba(240,240,240,0.85)" if dark else "rgba(0,0,0,0.7)"
    halo_color = "#fde047" if dark else "gold"  # brighter yellow on dark

    # Per-segment arrow annotations: each shaft+head shows direction of
    # conversation flow from turn i to turn i+1. `standoff` keeps the head
    # off the marker so the cluster color stays readable.
    for i in range(1, len(path_xy)):
        fig.add_annotation(
            x=float(path_xy[i, 0]),
            y=float(path_xy[i, 1]),
            ax=float(path_xy[i - 1, 0]),
            ay=float(path_xy[i - 1, 1]),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.4,
            arrowcolor=line_color,
            standoff=7,
            startstandoff=5,
            opacity=0.9,
        )
    # markers — cluster-colored fill, gold halo to pop them above the dim layer
    marker_colors = [cluster_palette.get(int(c), "#999999") for c in path_clusters]
    is_focus = path_turns == focus_turn
    sizes = np.where(is_focus, 16, 10)
    symbols = np.where(is_focus, "star", "circle")
    # 8-tuple customdata mirrors the base scatter so a click on a path marker
    # re-focuses on that turn instead of crashing the cd[1]/cd[2]/cd[5] reads.
    overlay_customdata = [
        [-1, run_id, int(t), "", "", int(c), "", ""]
        for t, c in zip(path_turns.tolist(), path_clusters.tolist())
    ]
    fig.add_trace(
        go.Scatter(
            x=path_xy[:, 0],
            y=path_xy[:, 1],
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=marker_colors,
                symbol=symbols,
                line=dict(width=2.0, color=halo_color),
            ),
            text=[
                str(int(t)) if (int(t) % 5 == 0 or t == focus_turn) else ""
                for t in path_turns
            ],
            textposition="top center",
            textfont=dict(size=9, color=text_color),
            hovertemplate=(
                "turn %{customdata[2]}<br>cid=%{customdata[5]}<extra>run path</extra>"
            ),
            customdata=overlay_customdata,
            name=f"path · run {run_id[:6]} · focus cid={focus_cluster}",
            showlegend=True,
        )
    )


# --- transcript rendering -------------------------------------------------


def _cluster_swatch_html(cid: int | None, color: str, dark: bool = False) -> str:
    """Tiny colored square + cid label for inline message annotation."""
    label_color = "#cccccc" if dark else "#444444"
    border = "rgba(255,255,255,0.4)" if dark else "rgba(0,0,0,0.35)"
    if cid is None:
        none_label = "#888888" if dark else "#888888"
        none_bg = "#444444" if dark else "#dddddd"
        return (
            f"<span style='display:inline-block;width:11px;height:11px;"
            f"background:{none_bg};border:1px solid {border};border-radius:2px;"
            f"vertical-align:middle;margin-right:6px;'></span>"
            f"<span style='font-size:11px;color:{none_label};'>cid=—</span>"
        )
    return (
        f"<span style='display:inline-block;width:11px;height:11px;"
        f"background:{color};border:1px solid {border};"
        f"border-radius:2px;vertical-align:middle;margin-right:6px;'></span>"
        f"<span style='font-size:11px;color:{label_color};'><b>cid={cid}</b></span>"
    )


def render_transcript(
    path: Path,
    focus_turn: int | None,
    cluster_by_turn: dict[int, int] | None = None,
    cluster_palette: dict[int, str] | None = None,
    focus_cluster: int | None = None,
    dark: bool = False,
) -> None:
    if not path.exists():
        st.warning(f"transcript not found: {path}")
        return
    lines = path.read_text().splitlines()
    if not lines:
        st.info("(empty transcript)")
        return

    header = json.loads(lines[0])
    st.markdown(
        f"### {header['model_variant']} / `{header['seed_name']}` · run "
        f"`{header['run_id'][:8]}`  ·  focus turn = "
        f"{focus_turn if focus_turn is not None else '—'}"
    )
    st.caption(
        f"model_tag: `{header['model_tag']}` · n_turns target: "
        f"{header['n_turns']} · started: {header['started_at']}"
    )
    with st.expander("seed prompt", expanded=False):
        st.markdown(f"_{header['seed_prompt']}_")

    st.markdown("---")

    palette = cluster_palette or {}
    cbt = cluster_by_turn or {}

    footer: dict | None = None
    for line in lines[1:]:
        rec = json.loads(line)
        if rec.get("turn_index") == -2:
            footer = rec
            continue
        ti = int(rec["turn_index"])
        agent = rec["agent"]
        is_focus = focus_turn is not None and ti == focus_turn
        cid = cbt.get(ti)
        color = palette.get(int(cid), "#dddddd") if cid is not None else "#dddddd"
        in_focus_cluster = focus_cluster is not None and cid == focus_cluster

        meta = (
            f"turn {ti:02d} · agent {agent} · {rec['elapsed_ms']} ms"
            + ("  ←  selected" if is_focus else "")
            + ("  · in focus cluster ★" if in_focus_cluster and not is_focus else "")
        )
        swatch = _cluster_swatch_html(cid, color, dark=dark)
        meta_color = "#aaaaaa" if dark else "#666666"
        prefix = "🎯 " if is_focus else ""
        # Visual anchor for the focused turn: a colored left border + bg tint
        # makes it stand out, and an HTML id lets us scroll-to it from JS.
        focus_outer_open = focus_outer_close = ""
        anchor = ""
        if is_focus:
            focus_bg = "rgba(255,224,102,0.12)" if dark else "rgba(255,210,80,0.18)"
            focus_border = "#fde047" if dark else "#f59e0b"
            anchor = "<span id='cluster-lab-focus-turn'></span>"
            focus_outer_open = (
                f"<div style='border-left:4px solid {focus_border};"
                f"background:{focus_bg};padding:8px 10px;border-radius:4px;"
                f"margin:-2px -4px 4px -4px;'>"
            )
            focus_outer_close = "</div>"

        with st.chat_message("user" if agent == "A" else "assistant"):
            st.markdown(
                f"{anchor}{focus_outer_open}"
                f"<div style='font-size:12px;color:{meta_color};margin-bottom:2px;'>"
                f"{prefix}{swatch}<span style='margin-left:6px;'>{meta}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if is_focus:
                st.markdown(f"**{rec['content']}**")
            else:
                st.markdown(rec["content"])
            if focus_outer_close:
                st.markdown(focus_outer_close, unsafe_allow_html=True)

    st.markdown("---")
    if footer:
        st.markdown(
            f"**stop_reason:** `{footer['stop_reason']}` · completed_turns: "
            f"{footer['completed_turns']} / {header['n_turns']} · finished: "
            f"{footer['finished_at']}"
        )

    # Scroll the focus turn into view inside the scrollable container.
    # `scrollIntoView` walks up to the nearest scrollable ancestor, which is
    # the st.container(height=...) the transcript lives in. The component
    # iframe is same-origin so it can reach window.parent.document.
    if focus_turn is not None:
        import streamlit.components.v1 as components

        components.html(
            f"""
            <script>
            (function() {{
              const tryScroll = (attempt) => {{
                const doc = window.parent.document;
                const tgt = doc.getElementById('cluster-lab-focus-turn');
                if (tgt) {{
                  tgt.scrollIntoView({{block: 'center', behavior: 'smooth'}});
                }} else if (attempt < 10) {{
                  setTimeout(() => tryScroll(attempt + 1), 80);
                }}
              }};
              tryScroll(0);
            }})();
            </script>
            """,
            height=0,
        )


# --- page -----------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="cluster lab", layout="wide")
    st.title("cluster lab — per-message KMeans")

    with st.sidebar:
        st.header("data")
        npz_path = st.text_input("emb_msgs npz", "artifacts/emb_msgs.npz")
        transcript_dir = st.text_input("transcripts dir", "transcripts/")
        vet_path = st.text_input(
            "vet_results jsonl (llama-guard)",
            "artifacts/vet_results.jsonl",
            help="Optional. Adds p_unsafe / verdict color-by options.",
        )

        st.header("filter")
        completed_only = st.checkbox("completed-only", value=True)
        min_chars = st.number_input(
            "min_chars", min_value=0, max_value=10000, value=150, step=10
        )

        st.header("balance (fit substrate)")
        balance = st.checkbox(
            "balance per (seed, variant)",
            value=True,
            help="Subsample so each (seed, variant) cell contributes the same number "
            "of points to the kmeans fit. Prevents class-imbalance artifacts.",
        )
        balance_target_override = st.number_input(
            "target_n per cell (0 = smallest cell)",
            min_value=0, max_value=20000, value=0, step=10,
            disabled=not balance,
        )

        st.header("kmeans")
        k = st.slider("k", min_value=2, max_value=100, value=8)
        kmeans_seed = st.number_input(
            "kmeans seed", min_value=0, max_value=10_000, value=0
        )

        st.header("projection")
        proj_method = st.radio(
            "method", ("PCA", "t-SNE"), horizontal=True, key="proj_method"
        )
        if proj_method == "t-SNE":
            tsne_perplexity = st.number_input(
                "perplexity", min_value=5.0, max_value=200.0, value=30.0, step=5.0
            )
            tsne_subsample = st.number_input(
                "max points (random subsample for speed; 0 = all)",
                min_value=0, max_value=200_000, value=8000, step=500,
            )
        else:
            tsne_perplexity = 30.0
            tsne_subsample = 0

        st.header("display")
        color_by = st.selectbox(
            "color by",
            (
                "cluster",
                "variant",
                "seed",
                "turn_index",
                "p_unsafe",
                "run_max_p_unsafe",
                "verdict_msg",
                "verdict_run",
            ),
            help="p_unsafe / verdict_* require a vet_results jsonl above.",
        )
        hover_chars = st.slider(
            "hover preview length", min_value=80, max_value=400, value=160, step=20
        )

    if not Path(npz_path).exists():
        st.error(f"npz not found: {npz_path}")
        return

    raw = load_npz(npz_path)
    n0 = len(raw["vectors"])

    # --- filter chain ---
    keep = np.ones(n0, dtype=bool)
    if completed_only:
        keep &= raw["stop_reasons"] == "completed"

    if int(min_chars) > 0:
        rids_all = tuple(raw["run_ids"].tolist())
        ti_all = tuple(int(x) for x in raw["turn_index"].tolist())
        lengths = cached_lengths(transcript_dir, rids_all, ti_all)
        keep &= lengths >= int(min_chars)

    vectors = raw["vectors"][keep]
    turn_idx = raw["turn_index"][keep]
    variants = raw["variants"][keep]
    seeds_arr = raw["seeds"][keep]
    run_ids = raw["run_ids"][keep]
    agents = raw["agents"][keep]

    if len(vectors) < max(k, 5):
        st.warning(f"only {len(vectors)} points after filtering; loosen filters.")
        return

    # --- llama-guard vet results: join per (run_id, turn_idx) ---
    vet = load_vet_results(vet_path, transcript_dir)
    p_unsafe_arr = np.array(
        [
            vet["per_msg_p"].get((rid, int(ti)), np.nan)
            for rid, ti in zip(run_ids, turn_idx)
        ],
        dtype=np.float64,
    )
    verdict_msg_arr = np.array(
        [
            vet["per_msg_v"].get((rid, int(ti)), "unknown")
            for rid, ti in zip(run_ids, turn_idx)
        ],
        dtype=str,
    )
    run_max_p_unsafe_arr = np.array(
        [vet["per_run"].get(rid, {}).get("max_p_unsafe", np.nan) for rid in run_ids],
        dtype=np.float64,
    )
    verdict_run_arr = np.array(
        [vet["per_run"].get(rid, {}).get("verdict", "unknown") for rid in run_ids],
        dtype=str,
    )

    # --- balance & fit ---
    cell_counts = Counter(zip(seeds_arr.tolist(), variants.tolist()))
    min_cell = min(cell_counts.values()) if cell_counts else 0
    target_n_eff: int | None = None
    if balance:
        target_n_eff = (
            int(balance_target_override) if balance_target_override > 0 else min_cell
        )

    labels, balanced_idx, centroids = fit_balanced_assign_full(
        vectors, seeds_arr, variants, target_n_eff, balance, k, int(kmeans_seed)
    )

    # --- projection ---
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    if proj_method == "PCA":
        proj, evr, pca_mean, pca_components = pca_2d(vectors)
        x_label = f"PC1 ({evr[0]:.1%})"
        y_label = f"PC2 ({evr[1]:.1%})"
        plot_idx = np.arange(len(vectors), dtype=np.int64)
    else:
        if tsne_subsample > 0 and tsne_subsample < len(vectors):
            rng = np.random.default_rng(int(kmeans_seed))
            plot_idx = np.sort(
                rng.choice(len(vectors), size=int(tsne_subsample), replace=False)
            )
        else:
            plot_idx = np.arange(len(vectors), dtype=np.int64)
        proj_sub = tsne_2d(
            vectors[plot_idx], float(tsne_perplexity), int(kmeans_seed)
        )
        proj = np.zeros((len(vectors), 2), dtype=np.float32)
        proj[plot_idx] = proj_sub
        x_label = f"t-SNE 1 (n_plot={len(plot_idx)}, perp={tsne_perplexity:g})"
        y_label = "t-SNE 2"

    # --- text previews for hover + click ---
    rids_kept = tuple(run_ids.tolist())
    ti_kept = tuple(int(x) for x in turn_idx.tolist())
    texts_full = cached_texts(transcript_dir, rids_kept, ti_kept)

    def _preview(t: str, n: int) -> str:
        s = " ".join(t.split())
        return s[:n] + ("…" if len(s) > n else "")

    previews = [_preview(t, hover_chars) for t in texts_full]

    # --- summary table ---
    summary = cluster_summary_df(labels, variants, seeds_arr, turn_idx)
    summary_sorted = summary.sort_values("JB%", ascending=False).reset_index(drop=True)

    base_jb = float((variants == "jailbroken").mean())
    n_full = len(vectors)
    st.markdown(
        f"**substrate** n={n_full} (from {n0} pre-filter) · "
        f"global JB={base_jb:.1%} · **fit** n={len(balanced_idx)} "
        f"({'balanced @ ' + str(target_n_eff) + '/cell' if balance else 'unbalanced (raw)'})"
    )

    cluster_ids_all = sorted({int(c) for c in np.unique(labels)})
    # Canonical palette — single source of truth across summary table,
    # scatter, trajectory, and path overlay.
    cluster_palette = cluster_color_map(cluster_ids_all)
    cluster_palette_str = {str(c): cluster_palette[c] for c in cluster_ids_all}

    def _bg_for_cid(s: pd.Series) -> list[str]:
        return [
            f"background-color: {cluster_palette.get(int(v), '#cccccc')}; "
            f"color: black; font-weight: bold"
            for v in s
        ]

    with st.expander("cluster summary (sorted by JB%)", expanded=True):
        st.dataframe(
            summary_sorted.style.format({
                "frac": "{:.1%}",
                "JB%": "{:.1%}",
                "enrich": "{:.2f}",
                "turn_μ": "{:.1f}",
                "turn_med": "{:.1f}",
            }).apply(_bg_for_cid, subset=["cid"]),  # type: ignore[attr-defined]
            height=min(36 * (len(summary_sorted) + 1) + 4, 360),
            width="stretch",
        )

    visible_clusters = st.multiselect(
        "show clusters (empty = all)",
        cluster_ids_all,
        default=[],
        format_func=lambda c: f"cid={c} (n={int((labels == c).sum())})",
    )

    # --- df for plotting ---
    df = pd.DataFrame(
        {
            "x": proj[:, 0],
            "y": proj[:, 1],
            "cluster": labels.astype(str),
            "variant": variants,
            "seed": seeds_arr,
            "turn_index": turn_idx,
            "agent": agents,
            "run_id": run_ids,
            "preview": previews,
            "row": np.arange(len(vectors), dtype=np.int64),
            "p_unsafe": p_unsafe_arr,
            "run_max_p_unsafe": run_max_p_unsafe_arr,
            "verdict_msg": verdict_msg_arr,
            "verdict_run": verdict_run_arr,
        }
    )

    show_mask = np.ones(len(df), dtype=bool)
    if proj_method == "t-SNE":
        # only points actually projected
        plotted = np.zeros(len(df), dtype=bool)
        plotted[plot_idx] = True
        show_mask &= plotted
    if visible_clusters:
        show_mask &= np.isin(labels, visible_clusters)
    df_show = df[show_mask].reset_index(drop=True)

    # --- color choice ---
    safety_palette = {
        "safe": "#16a34a",
        "unsafe": "#dc2626",
        "unknown": "#888888",
    }
    if color_by in ("turn_index", "p_unsafe", "run_max_p_unsafe"):
        # Continuous: turn_index uses Viridis (sequential, neutral); the
        # p_unsafe-style metrics use Inferno so high-risk pops red/yellow.
        scale = "Viridis" if color_by == "turn_index" else "Inferno"
        color_kwargs: dict = {
            "color": color_by,
            "color_continuous_scale": scale,
        }
        category_orders = None
    elif color_by in ("verdict_msg", "verdict_run"):
        color_kwargs = {
            "color": color_by,
            "color_discrete_map": safety_palette,
        }
        category_orders = {color_by: ["safe", "unsafe", "unknown"]}
    else:
        color_kwargs = {"color": color_by}
        if color_by == "cluster":
            order = [str(c) for c in cluster_ids_all]
            category_orders = {"cluster": order}
            color_kwargs["color_discrete_map"] = cluster_palette_str
        else:
            category_orders = None

    fig = px.scatter(
        df_show,
        x="x",
        y="y",
        custom_data=["row", "run_id", "turn_index", "variant", "seed",
                     "cluster", "agent", "preview"],
        category_orders=category_orders,
        height=620,
        **color_kwargs,
    )
    fig.update_traces(
        marker=dict(size=5, line=dict(width=0.2, color="rgba(0,0,0,0.4)")),
        unselected=dict(marker=dict(opacity=0.12)),
        selected=dict(marker=dict(opacity=1.0)),
        hovertemplate=(
            "<b>%{customdata[3]} | %{customdata[4]}</b> · "
            "agent %{customdata[6]} · turn %{customdata[2]}<br>"
            "cluster: %{customdata[5]}<br>"
            "run: %{customdata[1]}<br>"
            "<br>%{customdata[7]}<extra></extra>"
        ),
        selector=dict(type="scatter"),
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
        legend_title_text=color_by,
    )

    # --- selection state: session_state[chart_key] is the single source of
    # truth (more reliable than the return value of st.plotly_chart, which
    # comes back empty when we mutate fig before re-render).
    chart_key = f"scatter_{proj_method}_{k}_{kmeans_seed}_{int(balance)}"
    prior_event = st.session_state.get(chart_key)
    prior_sel: dict | None = None
    if isinstance(prior_event, dict):
        pts = prior_event.get("selection", {}).get("points", []) or []
        if pts:
            prior_sel = pts[0]

    n_base_traces = len(fig.data)  # type: ignore[arg-type]
    dark = _is_dark_theme()

    if prior_sel is not None:
        cd = prior_sel.get("customdata") or []
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) >= 6:
            sel_run_id = cd[1]
            sel_focus_turn = int(cd[2])
            sel_focus_cluster = int(cd[5])
            sel_curve = int(prior_sel.get("curve_number", -1))
            sel_point = int(prior_sel.get("point_number", -1))

            # Force the dim effect on base traces by setting selectedpoints
            # explicitly. selectedpoints=[] dims the whole trace; setting
            # [point] on the right trace highlights that one point.
            for i, trace in enumerate(fig.data):
                if i == sel_curve and sel_point >= 0:
                    trace.selectedpoints = [sel_point]  # type: ignore[attr-defined]
                else:
                    trace.selectedpoints = []  # type: ignore[attr-defined]

            _add_path_overlay(
                fig,
                raw=raw,
                run_id=sel_run_id,
                focus_turn=sel_focus_turn,
                focus_cluster=sel_focus_cluster,
                proj_method=proj_method,
                run_ids_plot=run_ids,
                turn_idx_plot=turn_idx,
                proj_plot=proj,
                plot_idx=plot_idx,
                centroids=centroids,
                cluster_palette=cluster_palette,
                pca_mean=pca_mean,
                pca_components=pca_components,
                dark=dark,
            )

            # Mark the overlay traces (line + markers) as fully "selected" so
            # they don't inherit the dim treatment we applied to base traces.
            for trace in fig.data[n_base_traces:]:
                xs = getattr(trace, "x", None)
                n_pts = 0 if xs is None else len(xs)
                trace.selectedpoints = list(range(n_pts))  # type: ignore[attr-defined]

    col_plot, col_panel = st.columns([3, 2], gap="large")

    with col_plot:
        st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode=("points",),
            key=chart_key,
        )

    path_index = index_transcripts(transcript_dir)

    with col_panel:
        header_l, header_r = st.columns([3, 1])
        with header_l:
            st.subheader("trajectory + transcript")
        with header_r:
            if prior_sel is not None and st.button(
                "✕ clear", key=f"clear_{chart_key}", help="exit trajectory mode",
                width="stretch",
            ):
                # Drop the cached selection event so the next rerun has none.
                st.session_state.pop(chart_key, None)
                st.rerun()
        if prior_sel is None:
            st.info("Click a point to load its trajectory + transcript.")
            return
        cd = prior_sel.get("customdata") or []
        if not isinstance(cd, (list, tuple, np.ndarray)) or len(cd) < 6:
            st.warning("could not resolve selection")
            return
        run_id = cd[1]
        focus_turn = int(cd[2])
        focus_cluster = int(cd[5])

        traj = run_trajectory(raw, run_id, centroids)
        if traj["turn_index"].size == 0:
            st.warning(f"no embedded turns found for run_id {run_id[:8]}")
        else:
            st.plotly_chart(
                trajectory_figure(traj, focus_cluster, focus_turn, cluster_palette),
                width="stretch",
                key=f"traj_{run_id}_{focus_cluster}_{focus_turn}",
            )
            in_focus = int((traj["assigned"] == focus_cluster).sum())
            st.caption(
                f"run `{run_id[:8]}` · {len(traj['turn_index'])} turns embedded · "
                f"{in_focus} assigned to cid={focus_cluster} "
                f"({in_focus / max(len(traj['turn_index']), 1):.0%}) · "
                f"clicked turn = {focus_turn}"
            )

        path = path_index.get(run_id)
        if not path:
            st.warning(f"no transcript file found for run_id {run_id[:8]}")
            return

        # Build {turn_index -> cid} from the trajectory's nearest-centroid
        # assignment so each rendered message can show its cluster swatch.
        cluster_by_turn = {
            int(t): int(c)
            for t, c in zip(traj["turn_index"].tolist(), traj["assigned"].tolist())
        }

        with st.container(height=560):
            render_transcript(
                Path(path),
                focus_turn,
                cluster_by_turn=cluster_by_turn,
                cluster_palette=cluster_palette,
                focus_cluster=focus_cluster,
                dark=dark,
            )


if __name__ == "__main__":
    main()
