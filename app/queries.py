"""Every LanceDB read the app makes, as a list of query-builder calls.

A query is a list of :class:`Step` (method name and arguments). :func:`run_steps`
makes exactly those calls on a table and :func:`format_steps` prints them as Python,
so the Debug panels show the query that actually ran. With a :data:`Trace`, each
query is recorded before it runs, failed ones included.

The app never writes: tables and indexes come from ``scripts/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from scripts.common import DISTANCE_TYPE, IMAGE, VIDEO, sql_string

VECTOR_PREVIEW = 4
FORMAT_WIDTH = 96
DEFAULT_LIMIT = 12
DEFAULT_NPROBES = 3
DEFAULT_REFINE = 5

# Columns the gallery and dialogs need; the full media bytes are read only for the dialog.
MEDIA_COLUMNS = ["media_id", "media_type", "filename", "dish", "duration_s", "frames_embedded", "thumbnail"]
FRAME_COLUMNS = ["frame_id", "video_id", "frame_index", "timestamp_s", "thumbnail"]


@dataclass(frozen=True)
class Arg:
    """A call argument that prints as ``text`` rather than as its value's repr."""

    value: object
    text: str
    comment: str = ""


@dataclass(frozen=True)
class Step:
    method: str
    args: tuple[object, ...] = ()
    kwargs: tuple[tuple[str, object], ...] = ()


@dataclass
class TracedQuery:
    label: str
    steps: tuple[Step, ...]
    root: str

    def text(self) -> str:
        return format_steps(self.steps, self.root)


Trace = list[TracedQuery]


def vector_arg(vector: Sequence[float], source: str) -> Arg:
    values = [float(x) for x in vector]
    preview = ", ".join(f"{v:.4f}" for v in values[:VECTOR_PREVIEW])
    return Arg(np.asarray(values, dtype=np.float32), f"[{preview}, ...]", f"{len(values)}-d {source}")


def _value(arg: object) -> object:
    return arg.value if isinstance(arg, Arg) else arg


def run_steps(target: object, steps: Sequence[Step]) -> Any:
    builder: Any = target
    for step in steps:
        builder = getattr(builder, step.method)(
            *(_value(a) for a in step.args), **{k: _value(v) for k, v in step.kwargs}
        )
    return builder


def _show(arg: object) -> str:
    return arg.text if isinstance(arg, Arg) else repr(arg)


def _format_step(step: Step) -> list[str]:
    parts = [(_show(a), a) for a in step.args] + [(f"{k}={_show(v)}", v) for k, v in step.kwargs]
    comments = [a.comment for _, a in parts if isinstance(a, Arg) and a.comment]
    line = f"  .{step.method}({', '.join(t for t, _ in parts)})"
    if not comments and len(line) <= FORMAT_WIDTH:
        return [line]
    if len(parts) == 1 and len(line) <= FORMAT_WIDTH:
        return [f"{line}  # {comments[0]}"]
    lines = [f"  .{step.method}("]
    for text, arg in parts:
        comment = arg.comment if isinstance(arg, Arg) else ""
        lines.append(f"      {text}," + (f"  # {comment}" if comment else ""))
    return [*lines, "  )"]


def format_steps(steps: Sequence[Step], root: str) -> str:
    lines = [root]
    for step in steps:
        lines.extend(_format_step(step))
    return "\n".join(lines)


def execute(table: Any, label: str, steps: list[Step], trace: Trace | None) -> Any:
    """Record the query in ``trace`` (if any), then run it on ``table``."""
    query = TracedQuery(label, tuple(steps), f"db.open_table({table.name!r})")
    if trace is not None:
        trace.append(query)
    return run_steps(table, steps)


# ── Search options ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnnOptions:
    """How a vector search over an indexed table runs."""

    exact: bool = False  # bypass the index: brute-force scan, the ground truth
    nprobes: int = DEFAULT_NPROBES
    refine_factor: int = DEFAULT_REFINE

    def steps(self) -> list[Step]:
        if self.exact:
            return [Step("bypass_vector_index")]
        return [Step("nprobes", (self.nprobes,)), Step("refine_factor", (self.refine_factor,))]


# ── Lookups ───────────────────────────────────────────────────────────────────


def media_row(table: Any, media_id: str, *, full: bool = False, trace: Trace | None = None) -> dict | None:
    """One ``media`` row by id: the gallery columns plus its vector (and the file bytes if ``full``)."""
    columns = [*MEDIA_COLUMNS, "embedding"] + (["media", "width", "height", "fps", "file_size_bytes"] if full else [])
    steps = [
        Step("search"),
        Step("where", (f"media_id = {sql_string(media_id)}",)),
        Step("select", (columns,)),
        Step("limit", (1,)),
        Step("to_pandas"),
    ]
    found = cast("pd.DataFrame", execute(table, f"{media_id} by id", steps, trace))
    return None if found.empty else found.iloc[0].to_dict()


def media_list(table: Any, media_type: str, *, trace: Trace | None = None) -> pd.DataFrame:
    """Ids (and dish names) of every video or reference image, for the pickers."""
    steps = [
        Step("search"),
        Step("where", (f"media_type = {sql_string(media_type)}",)),
        Step("select", (["media_id", "dish"],)),
        Step("limit", (1000,)),
        Step("to_pandas"),
    ]
    found = cast("pd.DataFrame", execute(table, f"List {media_type}s", steps, trace))
    return found.sort_values("media_id", key=lambda s: s.map(natural_key)).reset_index(drop=True)


def natural_key(name: str) -> tuple:
    """Sort ``download (2)`` before ``download (10)``."""
    digits = "".join(c for c in name if c.isdigit())
    return (name.split("(")[0], int(digits) if digits else -1, name)


def video_frames(table: Any, video_id: str, *, trace: Trace | None = None) -> pd.DataFrame:
    """A video's sampled frames in time order (thumbnails, no vectors)."""
    steps = [
        Step("search"),
        Step("where", (f"video_id = {sql_string(video_id)}",)),
        Step("select", (FRAME_COLUMNS,)),
        Step("limit", (1000,)),
        Step("to_pandas"),
    ]
    found = cast("pd.DataFrame", execute(table, f"Frames of {video_id}", steps, trace))
    return found.sort_values("frame_index").reset_index(drop=True)


# ── Vector searches ───────────────────────────────────────────────────────────


def media_search(
    table: Any,
    vector: Sequence[float],
    source: str,
    *,
    media_type: str | None,
    limit: int,
    exclude: str | None = None,
    trace: Trace | None = None,
) -> pd.DataFrame:
    """Nearest videos or reference images (``media`` is 39 rows: an exact scan, no index)."""
    where = []
    if media_type:
        where.append(f"media_type = {sql_string(media_type)}")
    if exclude:
        where.append(f"media_id != {sql_string(exclude)}")
    steps = [
        Step("search", (vector_arg(vector, source),), (("vector_column_name", "embedding"),)),
        Step("distance_type", (DISTANCE_TYPE,)),
    ]
    if where:
        steps.append(Step("where", (" AND ".join(where),), (("prefilter", True),)))
    steps += [Step("select", ([*MEDIA_COLUMNS, "_distance"],)), Step("limit", (limit,)), Step("to_pandas")]
    label = f"Nearest {media_type or 'media'}s to {source}"
    return with_similarity(cast("pd.DataFrame", execute(table, label, steps, trace)))


def frame_search(
    table: Any,
    vector: Sequence[float],
    source: str,
    *,
    limit: int,
    ann: AnnOptions,
    video_id: str | None = None,
    trace: Trace | None = None,
) -> pd.DataFrame:
    """Nearest video frames, optionally within one video (a BTree-indexed prefilter)."""
    steps = [
        Step("search", (vector_arg(vector, source),), (("vector_column_name", "embedding"),)),
        Step("distance_type", (DISTANCE_TYPE,)),
        *ann.steps(),
    ]
    if video_id:
        steps.append(Step("where", (f"video_id = {sql_string(video_id)}",), (("prefilter", True),)))
    steps += [Step("select", ([*FRAME_COLUMNS, "_distance"],)), Step("limit", (limit,)), Step("to_pandas")]
    scope = f" in {video_id}" if video_id else ""
    return with_similarity(cast("pd.DataFrame", execute(table, f"Nearest frames{scope} to {source}", steps, trace)))


def with_similarity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``similarity`` = 1 - cosine distance (the cosine similarity of the normalised vectors)."""
    if "_distance" in frame:
        frame = frame.assign(similarity=1.0 - frame["_distance"].astype(float))
    return frame


def best_frame_per_video(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per video: its best-matching frame."""
    if frames.empty:
        return frames
    best = frames.sort_values("similarity", ascending=False).drop_duplicates("video_id")
    return best.reset_index(drop=True)


# ── Reads for the Explore tab ─────────────────────────────────────────────────


def all_vectors(table: Any, columns: list[str], label: str, *, trace: Trace | None = None) -> pd.DataFrame:
    """Every row's vector and labels: a plain read with an explicit limit (remote-safe)."""
    steps = [Step("search"), Step("select", ([*columns, "embedding"],)), Step("limit", (100_000,)), Step("to_pandas")]
    return cast("pd.DataFrame", execute(table, label, steps, trace))


def nearest_dish(table: Any, vector: Sequence[float], source: str, *, trace: Trace | None = None) -> tuple[str, float]:
    """The reference dish nearest a vector (a cluster centroid), and its similarity."""
    found = media_search(table, vector, source, media_type=IMAGE, limit=1, trace=trace)
    if found.empty:
        return "", 0.0
    row = found.iloc[0]
    dish = row["dish"] if isinstance(row["dish"], str) else row["media_id"]
    return str(dish), float(row["similarity"])


__all__ = ["IMAGE", "VIDEO"]
