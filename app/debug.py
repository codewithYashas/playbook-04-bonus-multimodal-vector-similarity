"""The presenter's Debug section: raw rows of any table, with vectors and blobs summarised.

Reads go through ``table.search().select(columns).limit(n)``, which every table
supports (LanceDB Cloud included). Blob columns (the videos, images and thumbnails)
are left out unless asked for, so a glance at a table never pulls megabytes of video.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import pyarrow as pa

from queries import Step, Trace, execute

VECTOR_VALUES = 3


def is_vector(t: pa.DataType) -> bool:
    return pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type) and t.list_size >= 16


def is_blob(t: pa.DataType) -> bool:
    return pa.types.is_binary(t) or pa.types.is_large_binary(t)


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def blob_kind(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG"):
        return "PNG"
    if data[4:8] == b"ftyp":
        return "MP4"
    return "binary"


def summarize(data: pa.Table) -> pd.DataFrame:
    columns: dict[str, object] = {}
    for field, column in zip(data.schema, data.columns, strict=True):
        values = column.to_pylist()
        if is_vector(field.type):
            columns[field.name] = [
                None if v is None else f"{len(v)}-dim [{', '.join(f'{x:.3f}' for x in v[:VECTOR_VALUES])}, …]"
                for v in values
            ]
        elif is_blob(field.type):
            columns[field.name] = [None if v is None else f"{blob_kind(v)} · {human_size(len(v))}" for v in values]
        else:
            columns[field.name] = column.to_pandas()
    return pd.DataFrame(columns)


def raw_columns(schema: pa.Schema, *, blobs: bool) -> list[str]:
    return [f.name for f in schema if blobs or not is_blob(f.type)]


def read_raw(table: Any, rows: int, *, blobs: bool, trace: Trace | None) -> tuple[pd.DataFrame, list[str]]:
    """The first ``rows`` rows, summarised, and the blob columns left out."""
    columns = raw_columns(table.schema, blobs=blobs)
    steps = [Step("search"), Step("select", (columns,)), Step("limit", (rows,)), Step("to_arrow")]
    data = execute(table, f"Raw table: {table.name}", steps, trace)
    skipped = [] if blobs else [f.name for f in table.schema if is_blob(f.type)]
    return summarize(data), skipped


def index_rows(table: Any) -> list[dict[str, str]]:
    return [{"name": i.name, "type": str(i.index_type), "columns": ", ".join(i.columns)} for i in table.list_indices()]


def schema_rows(schema: pa.Schema) -> list[dict[str, str]]:
    return [{"column": f.name, "type": str(f.type)} for f in schema]


__all__: Sequence[str] = ["index_rows", "read_raw", "schema_rows", "summarize"]
