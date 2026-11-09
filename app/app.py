"""ByteEats vector similarity: food-prep videos and reference dishes in one LanceDB database.

Run with ``uv run streamlit run app/app.py`` after ``scripts/01_embed_media.py`` and
``scripts/02_create_indexes.py``. The page only opens tables and searches them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts  # noqa: E402
import debug  # noqa: E402
import explore  # noqa: E402
import queries as q  # noqa: E402
from scripts.common import (  # noqa: E402
    FRAMES_TABLE,
    IMAGE,
    MEDIA_TABLE,
    TABLES,
    VIDEO,
    Connection,
    connect_db,
    load_profiles,
    resolve_connection,
)

K = "byteeats."  # session-state key prefix
GALLERY_COLS = 4
NO_PROFILE = "(none)"
STATIC_CHART = {"displayModeBar": False}

DESCRIPTION = (
    "Food-prep videos and reference dish photos, embedded with ResNet-18 and stored with their "
    "vectors in LanceDB: two tables (`media`, `frames`) hold the files, thumbnails, metadata and "
    "embeddings together. Find similar dishes and videos, check a video against a reference dish "
    "frame by frame, and explore the embedding space."
)


@dataclass
class Options:
    limit: int
    ann: q.AnnOptions
    debug: bool
    trace: q.Trace | None = field(default=None)


# ── Connection ────────────────────────────────────────────────────────────────


def connect(conn: Connection) -> None:
    try:
        db = connect_db(conn)
        names = set(db.list_tables().tables)
        missing = [t for t in TABLES if t not in names]
        if missing:
            raise RuntimeError(
                f"{conn.uri} has no {', '.join(missing)} table. Run scripts/01_embed_media.py "
                "and scripts/02_create_indexes.py first."
            )
        st.session_state[K + "tables"] = {name: db.open_table(name) for name in TABLES}
        st.session_state[K + "reading"] = f"Reading {conn.describe()}"
        st.session_state.pop(K + "error", None)
    except Exception as e:  # shown on the page, so a bad URI doesn't crash it
        st.session_state.pop(K + "tables", None)
        st.session_state[K + "error"] = str(e)


def render_connection() -> None:
    profiles = load_profiles()
    if K + "uri" not in st.session_state:
        initial = resolve_connection()
        st.session_state[K + "uri"] = initial.uri
        st.session_state[K + "profile"] = initial.profile or NO_PROFILE
        connect(initial)
    with st.sidebar.expander("LanceDB connection", icon=":material/database:"):
        st.selectbox("Profile", [NO_PROFILE, *profiles], key=K + "profile", help="From ~/.lancedb/profiles.toml")
        st.text_input("URI", key=K + "uri", help="A local path, or db://… for LanceDB Cloud / Enterprise")
        st.caption(st.session_state.get(K + "reading", ""))
        st.caption("API keys come from the profile or LANCEDB_API_KEY and never reach the browser.")
        if st.button("Connect", width="stretch"):
            profile = st.session_state[K + "profile"]
            try:
                conn = resolve_connection(st.session_state[K + "uri"], None if profile == NO_PROFILE else profile)
            except ValueError as e:
                st.error(str(e))
                return
            connect(conn)
            for key in [k for k in st.session_state if k.startswith(K + "cache.")]:
                del st.session_state[key]


def render_sidebar() -> Options:
    render_connection()
    st.sidebar.subheader("Search")
    limit = st.sidebar.slider("Max results", 3, 24, 8, key=K + "limit")
    st.sidebar.caption(
        "`media` (39 rows) is always scanned exactly. `frames` (768 rows) has an IVF_HNSW_SQ index; "
        "these settings tune searches over it."
    )
    exact = st.sidebar.toggle(
        "Exact frame search (bypass the index)", key=K + "exact", help="bypass_vector_index(): a brute-force scan"
    )
    nprobes = st.sidebar.slider("nprobes", 1, 8, q.DEFAULT_NPROBES, disabled=exact, key=K + "nprobes")
    refine = st.sidebar.slider("refine_factor", 1, 20, q.DEFAULT_REFINE, disabled=exact, key=K + "refine")
    st.sidebar.divider()
    dbg = st.sidebar.toggle(
        ":material/build: Debug", key=K + "debug", help="Raw tables and the queries sent to LanceDB"
    )
    return Options(limit, q.AnnOptions(exact, nprobes, refine), dbg, [] if dbg else None)


# ── Small helpers ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading ResNet-18…")
def embedder():  # noqa: ANN201
    from scripts.media import Embedder

    return Embedder()


def label_for(row: dict | pd.Series) -> str:
    """A dish's name, else the media id (pandas reads a missing dish as NaN)."""
    dish = row.get("dish")
    return str(dish) if isinstance(dish, str) and dish else str(row.get("media_id"))


def show_query(opts: Options, start: int) -> None:
    """The queries sent since trace position ``start``, in a "LanceDB query" expander."""
    if not opts.debug or opts.trace is None:
        return
    with st.expander("LanceDB query", icon=":material/code:"):
        for traced in opts.trace[start:]:
            st.caption(traced.label)
            st.code(traced.text(), language="python")


def mark(opts: Options) -> int:
    return len(opts.trace) if opts.trace is not None else 0


def pickers(media: Any, opts: Options) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Video and dish lists, read once per connection."""
    if K + "cache.lists" not in st.session_state:
        st.session_state[K + "cache.lists"] = (
            q.media_list(media, VIDEO, trace=opts.trace),
            q.media_list(media, IMAGE, trace=opts.trace),
        )
    return st.session_state[K + "cache.lists"]


# ── Details dialog ────────────────────────────────────────────────────────────


@st.dialog("Details", width="large")
def details_dialog(tables: dict[str, Any], opts: Options) -> None:
    media, frames = tables[MEDIA_TABLE], tables[FRAMES_TABLE]
    active = st.session_state.get(K + "dialog_id")
    row = q.media_row(media, active, full=True, trace=opts.trace) if active else None
    if row is None:
        st.warning(f"{active} is not in the table.")
        return
    is_video = row["media_type"] == VIDEO
    left, right = st.columns([3, 2], gap="large")
    with left:
        if is_video:
            st.video(bytes(row["media"]), format="video/mp4", loop=True, autoplay=True, muted=True)
        else:
            st.image(bytes(row["media"]), width="stretch")
    with right:
        st.subheader(label_for(row))
        facts = {
            "Type": "video" if is_video else "reference dish photo",
            "File": row["filename"],
            "Size": f"{row['width']}×{row['height']}, {row['file_size_bytes'] / 1e6:.1f} MB",
        }
        if is_video:
            facts["Length"] = f"{row['duration_s']:.1f} s at {row['fps']:.0f} fps"
            facts["Embedding"] = f"mean of {row['frames_embedded']} frame vectors (every 5th frame)"
        else:
            facts["Embedding"] = "ResNet-18 of the photo"
        for key, value in facts.items():
            st.markdown(f"**{key}:** {value}")

    if is_video:
        strip = q.video_frames(frames, active, trace=opts.trace)
        picks = strip.iloc[np.linspace(0, len(strip) - 1, min(8, len(strip))).astype(int)] if len(strip) else strip
        st.caption(f"Sampled frames ({len(strip)} in `frames`, 8 shown)")
        cols = st.columns(max(1, len(picks)))
        for col, (_, f) in zip(cols, picks.iterrows(), strict=False):
            col.image(bytes(f["thumbnail"]), caption=f"{f['timestamp_s']:.1f}s", width="stretch")

    st.subheader("Nearest neighbours")
    start = mark(opts)
    neighbours = q.media_search(
        media,
        row["embedding"],
        f"embedding of {active}",
        media_type=None,
        limit=opts.limit,
        exclude=active,
        trace=opts.trace,
    )
    neighbours["label"] = [label_for(r) for _, r in neighbours.iterrows()]
    bars, grid = st.columns([1, 1], gap="large")
    with bars:
        st.plotly_chart(charts.neighbour_bars(neighbours), width="stretch", config=STATIC_CHART, key=K + "dialog.bars")
        st.caption(":orange[■] reference dish · :violet[■] video. Similarity is cosine, 1 = identical.")
    with grid:
        cols = st.columns(3)
        for i, (_, n) in enumerate(neighbours.head(9).iterrows()):
            with cols[i % 3]:
                st.image(bytes(n["thumbnail"]), width="stretch")
                if st.button(f"{n['label']} · {n['similarity']:.3f}", key=f"{K}dialog.nb.{i}", width="stretch"):
                    st.session_state[K + "dialog_id"] = n["media_id"]
                    st.rerun(scope="fragment")
    show_query(opts, start)

    points = media_map(media, opts)
    st.caption("Where it sits among all 39 videos and dishes (UMAP of the stored vectors). Click a point to open it.")
    event = st.plotly_chart(
        charts.map_figure(points, active=active, height=380),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=K + "dialog.map",
    )
    clicked = charts.selected_index(event)
    if clicked is not None and str(points.iloc[clicked]["id"]) != active:
        st.session_state[K + "dialog_id"] = str(points.iloc[clicked]["id"])
        st.rerun(scope="fragment")


def open_details(media_id: str, tables: dict[str, Any], opts: Options) -> None:
    st.session_state[K + "dialog_id"] = media_id
    st.session_state.pop(K + "dialog.map", None)
    details_dialog(tables, opts)


# ── Search tab ────────────────────────────────────────────────────────────────

FROM_DISH, FROM_VIDEO, FROM_UPLOAD = "Reference dish", "Video", "Upload a photo"
FIND_VIDEOS, FIND_DISHES, FIND_FRAMES = "Videos", "Reference dishes", "Video frames"


def render_search_tab(tables: dict[str, Any], opts: Options) -> None:
    media, frames = tables[MEDIA_TABLE], tables[FRAMES_TABLE]
    videos, dishes = pickers(media, opts)
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        source = st.segmented_control("Query", [FROM_DISH, FROM_VIDEO, FROM_UPLOAD], default=FROM_DISH, key=K + "src")
        start = mark(opts)
        vector, query_id, query_label, caption, thumb = None, None, "", "", None
        if source == FROM_UPLOAD:
            upload = st.file_uploader("A photo of a dish", type=["jpg", "jpeg", "png"], key=K + "upload")
            if upload is not None:
                from PIL import Image

                image = Image.open(upload).convert("RGB")
                vector = embedder().embed_image(image).tolist()
                query_label, caption, thumb = f"ResNet-18 of {upload.name}", upload.name, upload.getvalue()
        else:
            options = dishes if source == FROM_DISH else videos
            labels = {r["media_id"]: label_for(r) for _, r in options.iterrows()}
            query_id = st.selectbox(source, list(labels), format_func=labels.get, key=K + f"pick.{source}")
            row = q.media_row(media, query_id, trace=opts.trace) if query_id else None
            if row is not None:
                vector, query_label = row["embedding"], f"embedding of {query_id}"
                caption, thumb = labels[query_id], bytes(row["thumbnail"])
        target = st.segmented_control(
            "Find", [FIND_VIDEOS, FIND_DISHES, FIND_FRAMES], default=FIND_VIDEOS, key=K + "target"
        )
    with c2:
        if thumb:
            st.image(thumb, caption=caption, width=320)
    if vector is None or target is None:
        st.info("Pick a dish or a video, or upload a photo.", icon=":material/search:")
        return

    if target == FIND_FRAMES:
        results = q.frame_search(frames, vector, query_label, limit=opts.limit, ann=opts.ann, trace=opts.trace)
        caption = f"Top {len(results)} of 768 frames by cosine similarity"
    else:
        kind = VIDEO if target == FIND_VIDEOS else IMAGE
        results = q.media_search(
            media, vector, query_label, media_type=kind, limit=opts.limit, exclude=query_id, trace=opts.trace
        )
        caption = f"Top {len(results)} {target.lower()} by cosine similarity (exact scan of `media`)"
    st.subheader("Results")
    st.caption(caption)
    show_query(opts, start)
    render_gallery(results, tables, opts, frames_mode=target == FIND_FRAMES)


def render_gallery(results: pd.DataFrame, tables: dict[str, Any], opts: Options, *, frames_mode: bool) -> None:
    cols = st.columns(GALLERY_COLS, gap="medium")
    for i, (_, r) in enumerate(results.iterrows()):
        with cols[i % GALLERY_COLS], st.container(border=True):
            st.image(bytes(r["thumbnail"]), width="stretch")
            if frames_mode:
                st.markdown(f"**{r['video_id']}** at {r['timestamp_s']:.1f}s")
                target = r["video_id"]
            else:
                st.markdown(f"**{label_for(r)}**")
                target = r["media_id"]
            st.caption(f"similarity {r['similarity']:.4f}")
            if st.button("Details →", key=f"{K}details.{i}", width="stretch"):
                open_details(target, tables, opts)


# ── Quality check tab ─────────────────────────────────────────────────────────


def render_quality_tab(tables: dict[str, Any], opts: Options) -> None:
    media, frames = tables[MEDIA_TABLE], tables[FRAMES_TABLE]
    videos, dishes = pickers(media, opts)
    st.markdown(
        "Does a prep video show the dish it should? Each video's sampled frames are already in `frames`, "
        "so the check is one vector search: the reference photo's vector against that video's frames, "
        "prefiltered by `video_id`. The best frame's similarity is the score."
    )
    c1, c2, c3 = st.columns(3)
    dish_labels = {r["media_id"]: label_for(r) for _, r in dishes.iterrows()}
    dish_id = c1.selectbox("Reference dish", list(dish_labels), format_func=dish_labels.get, key=K + "qc.dish")
    video_id = c2.selectbox("Video", videos["media_id"].tolist(), key=K + "qc.video")
    threshold = c3.slider("Pass threshold", 0.50, 0.95, 0.75, 0.01, key=K + "qc.threshold")
    if not dish_id or not video_id:
        return

    start = mark(opts)
    ref = q.media_row(media, dish_id, trace=opts.trace)
    scored = q.frame_search(
        frames,
        ref["embedding"],
        f"embedding of {dish_id}",
        limit=1000,
        ann=opts.ann,
        video_id=video_id,
        trace=opts.trace,
    ).sort_values("frame_index")
    if scored.empty:
        st.warning(f"No frames for {video_id}.")
        return
    best = scored.loc[scored["similarity"].idxmax()]
    passed = best["similarity"] >= threshold

    m1, m2, m3 = st.columns(3)
    m1.metric("Best-frame similarity", f"{best['similarity']:.4f}")
    m2.metric("Threshold", f"{threshold:.2f}")
    m3.metric(
        "Result",
        "PASSED" if passed else "FAILED",
        "Acceptable" if passed else "Needs review",
        delta_color="normal" if passed else "inverse",
    )
    i1, i2, chart = st.columns([1, 1, 2], gap="medium")
    i1.image(bytes(ref["thumbnail"]), caption=f"Reference: {label_for(ref)}", width="stretch")
    i2.image(bytes(best["thumbnail"]), caption=f"Best frame: {video_id} at {best['timestamp_s']:.1f}s", width="stretch")
    with chart:
        st.plotly_chart(
            charts.timeline_figure(scored, threshold, int(best["frame_index"])),
            width="stretch",
            config=STATIC_CHART,
            key=K + "qc.timeline",
        )
    show_query(opts, start)

    st.subheader(f"Every video against {label_for(ref)}")
    st.caption("One search over all 768 frames, then each video's best frame. Green passes the threshold.")
    start = mark(opts)
    ranked = q.best_frame_per_video(
        q.frame_search(frames, ref["embedding"], f"embedding of {dish_id}", limit=1000, ann=opts.ann, trace=opts.trace)
    )
    n_pass = int((ranked["similarity"] >= threshold).sum())
    st.caption(f"**{n_pass}** of {len(ranked)} videos pass at {threshold:.2f}.")
    st.plotly_chart(
        charts.ranking_figure(ranked, threshold), width="stretch", config=STATIC_CHART, key=K + "qc.ranking"
    )
    show_query(opts, start)


# ── Explore tab ───────────────────────────────────────────────────────────────

SPACE_MEDIA, SPACE_FRAMES = "Videos and dishes (39)", "Video frames (768)"


@st.cache_data(show_spinner="Projecting with UMAP…")
def projected(vectors: np.ndarray) -> np.ndarray:
    return explore.project(vectors)


@st.cache_data(show_spinner="Clustering…")
def clustered(vectors: np.ndarray, algorithm: str, k: int, min_size: int) -> np.ndarray:
    return explore.cluster(vectors, algorithm, k, min_size)


def media_map(media: Any, opts: Options, labels: np.ndarray | None = None) -> pd.DataFrame:
    """All 39 media rows, projected to 2-D, with their cluster labels (all 0 when none given)."""
    if K + "cache.media_vectors" not in st.session_state:
        st.session_state[K + "cache.media_vectors"] = q.all_vectors(
            media, ["media_id", "media_type", "dish"], "All media vectors", trace=opts.trace
        )
    data = st.session_state[K + "cache.media_vectors"]
    xy = projected(explore.stack(data["embedding"]))
    return pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "id": data["media_id"],
            "label": [label_for(r) for _, r in data.iterrows()],
            "kind": data["media_type"],
            "cluster": labels if labels is not None else np.zeros(len(data), dtype=int),
        }
    )


def render_explore_tab(tables: dict[str, Any], opts: Options) -> None:
    media, frames = tables[MEDIA_TABLE], tables[FRAMES_TABLE]
    st.markdown(
        "The stored vectors, read straight from LanceDB, projected to 2-D with UMAP (cosine) and clustered "
        "in the full 512-d space. Stars are reference dishes. Each cluster is named after the reference dish "
        "nearest its centroid: a LanceDB search with the centroid as the query."
    )
    c1, c2, c3 = st.columns(3)
    space = c1.radio("Points", [SPACE_MEDIA, SPACE_FRAMES], key=K + "ex.space")
    algorithm = c2.radio("Clustering", explore.ALGORITHMS, horizontal=True, key=K + "ex.algo")
    k = c3.slider("K-Means k", 2, 10, 4, disabled=algorithm != explore.KMEANS, key=K + "ex.k")
    min_size = c3.slider("HDBSCAN min cluster size", 2, 30, 3, disabled=algorithm != explore.HDBSCAN, key=K + "ex.min")

    start = mark(opts)
    if space == SPACE_MEDIA:
        media_map(media, opts)  # loads and caches the vectors
        data = st.session_state[K + "cache.media_vectors"]
        ids, kinds = data["media_id"].astype(str).tolist(), data["media_type"].tolist()
        names = [label_for(r) for _, r in data.iterrows()]
    else:
        if K + "cache.frame_vectors" not in st.session_state:
            st.session_state[K + "cache.frame_vectors"] = q.all_vectors(
                frames, ["frame_id", "video_id", "timestamp_s"], "All frame vectors", trace=opts.trace
            )
        data = st.session_state[K + "cache.frame_vectors"]
        ids, kinds = data["video_id"].astype(str).tolist(), ["frame"] * len(data)
        names = [f"{v} at {t:.1f}s" for v, t in zip(data["video_id"], data["timestamp_s"], strict=True)]
    vectors = explore.stack(data["embedding"])
    labels = clustered(vectors, algorithm, k, min_size)
    xy = projected(vectors)
    points = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "id": ids, "label": names, "kind": kinds, "cluster": labels})

    event = st.plotly_chart(
        charts.map_figure(points), width="stretch", on_select="rerun", selection_mode="points", key=K + "ex.map"
    )
    st.caption(
        f"{len(points)} points. Click one to open its video or dish. The studio photos sit apart from the "
        "video frames (a modality gap: lighting, framing and on-screen overlays differ), which is why the "
        "quality check compares a dish with a video's best frame, not with the video's mean vector."
    )
    clicked = charts.selected_index(event)
    if clicked != st.session_state.get(K + "ex.last_click"):
        st.session_state[K + "ex.last_click"] = clicked
        if clicked is not None and 0 <= clicked < len(points):
            open_details(str(points.iloc[clicked]["id"]), tables, opts)

    rows = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        # Frames name their video; several frames of one video count once.
        member_names = sorted({ids[i] if space == SPACE_FRAMES else names[i] for i in members}, key=q.natural_key)
        row = {
            "cluster": "noise" if label == explore.NOISE else f"cluster {label}",
            "size": len(members),
            "nearest dish": "",
            "similarity": None,
            "cohesion": round(explore.cohesion(vectors[members]), 4),
            "members": ", ".join(member_names[:8]) + (" …" if len(member_names) > 8 else ""),
        }
        if label != explore.NOISE:
            dish, sim = q.nearest_dish(
                media, explore.centroid(vectors[members]), f"centroid of cluster {label}", trace=opts.trace
            )
            row.update({"nearest dish": dish, "similarity": round(sim, 4)})
        rows.append(row)
    table = pd.DataFrame(rows)
    st.subheader("Clusters")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "similarity": st.column_config.NumberColumn(help="Cosine similarity of the centroid to the nearest dish"),
            "cohesion": st.column_config.NumberColumn(help="Mean cosine similarity of members to their centroid"),
        },
    )
    with st.expander("Outliers: least like everything else"):
        st.dataframe(explore.outliers(vectors, names), width="stretch", hide_index=True)
    show_query(opts, start)


# ── Debug section ─────────────────────────────────────────────────────────────


def render_debug(tables: dict[str, Any], opts: Options) -> None:
    st.divider()
    st.subheader(":material/build: Debug")
    with st.expander("Raw table", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 1])
        name = c1.selectbox("Table", list(tables), key=K + "dbg.table")
        rows = c2.slider("Rows", 1, 200, 10, key=K + "dbg.rows")
        blobs = c3.checkbox("Include blob columns", key=K + "dbg.blobs", help="Videos, photos and thumbnails")
        frame, skipped = debug.read_raw(tables[name], rows, blobs=blobs, trace=opts.trace)
        st.dataframe(frame, width="stretch", hide_index=True)
        if skipped:
            st.caption(f"Not read: {', '.join(skipped)} (blob columns). Tick the box to include them.")
    with st.expander("Table stats"):
        for name, table in tables.items():
            st.markdown(f"**{name}**: {table.count_rows():,} rows, version {table.version}")
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame(debug.schema_rows(table.schema)), width="stretch", hide_index=True)
            c2.dataframe(pd.DataFrame(debug.index_rows(table)), width="stretch", hide_index=True)
    with st.expander(f"All queries this run ({len(opts.trace or [])})"):
        for traced in opts.trace or []:
            st.caption(traced.label)
            st.code(traced.text(), language="python")


# ── Page ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="ByteEats Vector Similarity",
        page_icon=":material/ramen_dining:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Hide Streamlit's Deploy button (the menu keeps Rerun), and a larger base font for presenting.
    st.html("<style>[data-testid='stAppDeployButton'] { display: none; } html { font-size: 20px; }</style>")
    st.title("ByteEats Vector Similarity")
    st.caption(DESCRIPTION)
    opts = render_sidebar()
    tables = st.session_state.get(K + "tables")
    if tables is None:
        st.error(st.session_state.get(K + "error", "Not connected."), icon=":material/error:")
        return

    search_tab, explore_tab, check_tab = st.tabs(["Search", "Explore", "Check"])
    with search_tab:
        render_search_tab(tables, opts)
    with explore_tab:
        render_explore_tab(tables, opts)
    with check_tab:
        render_quality_tab(tables, opts)
    if opts.debug:
        render_debug(tables, opts)


main()
