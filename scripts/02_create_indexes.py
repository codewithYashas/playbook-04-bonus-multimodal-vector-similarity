"""Create the indexes the app's queries use.

- ``frames.embedding``: an IVF_HNSW_SQ vector index (cosine). 768 frames is small, but
  it's the table the quality check and frame search scan, and the index is what a
  production-sized frame table would need. ``03_compare_indexes.py`` compares the types.
- ``frames.video_id``: a BTree scalar index, for the quality check's
  ``where("video_id = ...")`` prefilter and the details dialog's frame strip.
- ``media.media_type``: a Bitmap scalar index (two values: video, image).
- ``media.media_id``: a BTree scalar index, for lookups by id.

``media`` has 39 rows and no vector index: at that size an exact (flat) scan is both
the fastest and the most accurate search.

    uv run python scripts/02_create_indexes.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lancedb.index import Bitmap, BTree, IvfHnswSq  # noqa: E402

from scripts.common import (  # noqa: E402
    DISTANCE_TYPE,
    FRAMES_TABLE,
    MEDIA_TABLE,
    add_connection_args,
    connect_db,
    resolve_connection,
)

WAIT = timedelta(minutes=10)  # Cloud builds indexes in the background; local builds are synchronous


def frame_index_config(rows: int) -> IvfHnswSq:
    # About one partition per 256 frames: 3 for the sample data.
    return IvfHnswSq(distance_type=DISTANCE_TYPE, num_partitions=max(1, rows // 256))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_connection_args(parser)
    args = parser.parse_args()

    conn = resolve_connection(args.uri, args.profile)
    db = connect_db(conn)
    media, frames = db.open_table(MEDIA_TABLE), db.open_table(FRAMES_TABLE)

    config = frame_index_config(frames.count_rows())
    frames.create_index("embedding", config=config, replace=True, wait_timeout=WAIT)
    frames.create_index("video_id", config=BTree(), replace=True, wait_timeout=WAIT)
    media.create_index("media_type", config=Bitmap(), replace=True, wait_timeout=WAIT)
    media.create_index("media_id", config=BTree(), replace=True, wait_timeout=WAIT)

    print(f"Indexes at {conn.describe()}:")
    for table in (media, frames):
        for index in table.list_indices():
            print(f"  {table.name}.{', '.join(index.columns)}: {index.index_type} ({index.name})")


if __name__ == "__main__":
    main()
