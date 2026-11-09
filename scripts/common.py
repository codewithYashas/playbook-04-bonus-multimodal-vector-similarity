"""Paths, table schemas and the LanceDB connection, shared by the pipeline and the app.

Everything the demo stores lives in one LanceDB database, ``byteeats.lancedb/`` at the
repository root (``BYTEEATS_URI`` points elsewhere, for example a ``db://`` database on
LanceDB Cloud or Enterprise). It holds two tables:

- ``media``: one row per video and per reference image, with the original file bytes,
  a JPEG thumbnail, its metadata and its 512-d ResNet-18 embedding.
- ``frames``: one row per sampled video frame (every 5th frame), with a thumbnail and its
  embedding. A video's embedding in ``media`` is the normalised mean of its frames'.

Named connection profiles live in ``~/.lancedb/profiles.toml``, shared with the other
fe-demos. This demo reads a profile's ``byteeats_uri``, plus the shared ``api_key``,
``region`` and ``host_override``. ``LANCEDB_PROFILE`` names the profile to start on; blank
means none.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    import lancedb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
IMAGE_DIR = DATA_DIR / "images"
DEFAULT_DB = ROOT / "byteeats.lancedb"

MEDIA_TABLE = "media"
FRAMES_TABLE = "frames"
TABLES = (MEDIA_TABLE, FRAMES_TABLE)

VIDEO, IMAGE = "video", "image"
EMBEDDING_DIM = 512
# Cosine distance on L2-normalised ResNet-18 features; similarity = 1 - distance.
DISTANCE_TYPE = "cosine"
MODEL_NAME = "resnet18 (torchvision IMAGENET1K_V1, avgpool features)"

ENV_URI = "BYTEEATS_URI"
ENV_PROFILE = "LANCEDB_PROFILE"
ENV_API_KEY = "LANCEDB_API_KEY"
ENV_REGION = "LANCEDB_REGION"
ENV_HOST_OVERRIDE = "LANCEDB_HOST_OVERRIDE"
DEFAULT_PROFILE = "my-cluster"
PROFILE_URI_KEY = "byteeats_uri"
PROFILES_PATH = Path.home() / ".lancedb" / "profiles.toml"
DEFAULT_REGION = "us-east-1"


def vector_type(dim: int = EMBEDDING_DIM) -> pa.DataType:
    return pa.list_(pa.float32(), dim)


MEDIA_SCHEMA = pa.schema(
    [
        pa.field("media_id", pa.string(), nullable=False),
        pa.field("media_type", pa.string(), nullable=False),
        pa.field("filename", pa.string()),
        # The dish a reference image shows ("Tuna Lover"); empty for videos.
        pa.field("dish", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("duration_s", pa.float32()),
        pa.field("fps", pa.float32()),
        pa.field("frame_count", pa.int32()),
        pa.field("frames_embedded", pa.int32()),
        pa.field("file_size_bytes", pa.int64()),
        pa.field("media", pa.large_binary()),
        pa.field("thumbnail", pa.binary()),
        pa.field("embedding", vector_type()),
        pa.field("model", pa.string()),
        pa.field("embedded_at", pa.timestamp("us")),
    ]
)

FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_id", pa.string(), nullable=False),
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("frame_index", pa.int32()),
        pa.field("timestamp_s", pa.float32()),
        pa.field("thumbnail", pa.binary()),
        pa.field("embedding", vector_type()),
    ]
)


def sql_string(value: str) -> str:
    """Quote ``value`` as a SQL string literal, doubling single quotes (``Kid's_Crab_Bowl``)."""
    return "'" + value.replace("'", "''") + "'"


# ── Connection ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Connection:
    """Where to connect: a local path or a ``db://`` URI, plus Cloud credentials."""

    uri: str
    api_key: str | None = None
    region: str = DEFAULT_REGION
    host_override: str | None = None
    profile: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.uri.startswith("db://")

    def describe(self) -> str:
        where = "LanceDB Cloud/Enterprise" if self.is_remote else "local"
        via = f" (profile {self.profile})" if self.profile else ""
        return f"{where} · {self.uri}{via}"


def load_profiles(path: Path = PROFILES_PATH) -> dict[str, dict[str, object]]:
    """The profiles in ``~/.lancedb/profiles.toml``; empty when the file is missing."""
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return {name: dict(body) for name, body in data.items() if isinstance(body, dict)}


def chosen_profile(explicit: str | None = None) -> str | None:
    """The profile to use: ``explicit``, else ``LANCEDB_PROFILE``, else ``my-cluster`` if present.

    A blank ``LANCEDB_PROFILE`` means no profile.
    """
    if explicit:
        return explicit
    env = os.environ.get(ENV_PROFILE)
    if env is not None:
        return env.strip() or None
    return DEFAULT_PROFILE if DEFAULT_PROFILE in load_profiles() else None


def resolve_connection(uri: str | None = None, profile: str | None = None) -> Connection:
    """Settings from, in order: arguments, ``BYTEEATS_URI``, the profile, then the local default.

    A profile only supplies the URI when it has a ``byteeats_uri`` entry, so a
    credentials-only profile leaves the demo on its local database.
    """
    name = chosen_profile(profile)
    body = load_profiles().get(name, {}) if name else {}
    if profile and not body:
        msg = f"profile {profile!r} not found in {PROFILES_PATH}"
        raise ValueError(msg)
    profile_uri = body.get(PROFILE_URI_KEY)
    resolved = uri or os.environ.get(ENV_URI) or profile_uri or str(DEFAULT_DB)
    used_profile = name if body and (resolved == profile_uri or resolved.startswith("db://")) else None
    return Connection(
        uri=str(resolved),
        api_key=os.environ.get(ENV_API_KEY) or _str(body.get("api_key")),
        region=os.environ.get(ENV_REGION) or _str(body.get("region")) or DEFAULT_REGION,
        host_override=os.environ.get(ENV_HOST_OVERRIDE) or _str(body.get("host_override")),
        profile=used_profile,
    )


def _str(value: object) -> str | None:
    return str(value) if value else None


def connect_db(conn: Connection | None = None) -> lancedb.DBConnection:
    """Open the demo's database; the only place the demo calls ``lancedb.connect``."""
    import lancedb

    conn = conn or resolve_connection()
    if conn.is_remote:
        if not conn.api_key:
            msg = f"{conn.uri} needs an API key: set {ENV_API_KEY} or a profile's api_key"
            raise ValueError(msg)
        kwargs: dict[str, object] = {"api_key": conn.api_key, "region": conn.region}
        if conn.host_override:
            kwargs["host_override"] = conn.host_override
        return lancedb.connect(conn.uri, **kwargs)
    return lancedb.connect(conn.uri)


def add_connection_args(parser) -> None:  # noqa: ANN001 - argparse.ArgumentParser
    parser.add_argument("--uri", help=f"database URI (default: ${ENV_URI}, the profile, or {DEFAULT_DB.name})")
    parser.add_argument("--profile", help="profile in ~/.lancedb/profiles.toml")


# Remote tables take writes in requests capped at about 32 MiB; the videos travel inline.
MAX_WRITE_BYTES = 24 * 1024 * 1024


def write_table(db: lancedb.DBConnection, name: str, data: pa.Table) -> lancedb.table.Table:
    """Create (or replace) table ``name`` and add ``data`` in chunks of at most 24 MiB."""
    table = db.create_table(name, schema=data.schema, mode="overwrite")
    batch: list[pa.RecordBatch] = []
    size = 0
    for record_batch in data.to_batches(max_chunksize=1):
        if batch and size + record_batch.nbytes > MAX_WRITE_BYTES:
            table.add(pa.Table.from_batches(batch, data.schema))
            batch, size = [], 0
        batch.append(record_batch)
        size += record_batch.nbytes
    if batch:
        table.add(pa.Table.from_batches(batch, data.schema))
    return table
