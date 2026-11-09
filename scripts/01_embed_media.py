"""Embed the sample videos and reference images straight into LanceDB.

Writes two tables (see ``common.py``):

- ``media``: 32 videos + 7 reference images, with file bytes, thumbnail, metadata, embedding
- ``frames``: every 5th frame of each video (24 per clip), with thumbnail and embedding

No ``.npy`` or JSON sidecar files: the vectors live next to the data they describe.
Rerunning replaces both tables. About 2-3 minutes on a laptop CPU for all 32 videos.

    uv run python scripts/01_embed_media.py              # everything
    uv run python scripts/01_embed_media.py --limit 3    # 3 videos + all images (quick check)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (  # noqa: E402
    FRAMES_SCHEMA,
    FRAMES_TABLE,
    IMAGE,
    IMAGE_DIR,
    MEDIA_SCHEMA,
    MEDIA_TABLE,
    MODEL_NAME,
    VIDEO,
    VIDEO_DIR,
    add_connection_args,
    connect_db,
    resolve_connection,
    write_table,
)
from scripts.media import FRAME_THUMB_WIDTH, THUMB_WIDTH, Embedder, jpeg_bytes, read_frames  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def dish_name(stem: str) -> str:
    return stem.replace("_", " ")


def embed_videos(embedder: Embedder, paths: list[Path], now: datetime) -> tuple[list[dict], list[dict]]:
    media, frames = [], []
    for n, path in enumerate(paths, 1):
        start = time.perf_counter()
        clip = read_frames(str(path))
        if not clip.frames:
            print(f"  skipped {path.name}: no frames decoded")
            continue
        per_frame, video_vec = embedder.embed_frames(clip.frames)
        video_id = path.stem
        middle = clip.frames[len(clip.frames) // 2]
        media.append(
            {
                "media_id": video_id,
                "media_type": VIDEO,
                "filename": path.name,
                "dish": None,
                "width": clip.width,
                "height": clip.height,
                "duration_s": clip.duration_s,
                "fps": clip.fps,
                "frame_count": clip.frame_count,
                "frames_embedded": len(clip.frames),
                "file_size_bytes": path.stat().st_size,
                "media": path.read_bytes(),
                "thumbnail": jpeg_bytes(Image.fromarray(middle), THUMB_WIDTH),
                "embedding": video_vec.tolist(),
                "model": MODEL_NAME,
                "embedded_at": now,
            }
        )
        for frame, index, vec in zip(clip.frames, clip.indices, per_frame, strict=True):
            frames.append(
                {
                    "frame_id": f"{video_id}#{index:04d}",
                    "video_id": video_id,
                    "frame_index": index,
                    "timestamp_s": index / clip.fps if clip.fps else 0.0,
                    "thumbnail": jpeg_bytes(Image.fromarray(frame), FRAME_THUMB_WIDTH),
                    "embedding": vec.tolist(),
                }
            )
        print(f"  [{n}/{len(paths)}] {path.name}: {len(clip.frames)} frames in {time.perf_counter() - start:.1f}s")
    return media, frames


def embed_images(embedder: Embedder, paths: list[Path], now: datetime) -> list[dict]:
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        rows.append(
            {
                "media_id": path.stem,
                "media_type": IMAGE,
                "filename": path.name,
                "dish": dish_name(path.stem),
                "width": image.width,
                "height": image.height,
                "duration_s": None,
                "fps": None,
                "frame_count": None,
                "frames_embedded": None,
                "file_size_bytes": path.stat().st_size,
                "media": path.read_bytes(),
                "thumbnail": jpeg_bytes(image, THUMB_WIDTH),
                "embedding": embedder.embed_image(image).tolist(),
                "model": MODEL_NAME,
                "embedded_at": now,
            }
        )
        print(f"  {path.name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=VIDEO_DIR)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--limit", type=int, help="embed only the first N videos (sorted by name)")
    add_connection_args(parser)
    args = parser.parse_args()

    video_paths = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit:
        video_paths = video_paths[: args.limit]

    conn = resolve_connection(args.uri, args.profile)
    db = connect_db(conn)
    embedder = Embedder()
    now = datetime.now(UTC).replace(tzinfo=None)
    print(f"Embedding with {MODEL_NAME} on {embedder.device}")

    start = time.perf_counter()
    print(f"Videos ({len(video_paths)}):")
    media_rows, frame_rows = embed_videos(embedder, video_paths, now)
    print(f"Reference images ({len(image_paths)}):")
    media_rows += embed_images(embedder, image_paths, now)

    media = write_table(db, MEDIA_TABLE, pa.Table.from_pylist(media_rows, MEDIA_SCHEMA))
    frames = write_table(db, FRAMES_TABLE, pa.Table.from_pylist(frame_rows, FRAMES_SCHEMA))
    print(
        f"\nWrote {media.count_rows()} rows to '{MEDIA_TABLE}' and {frames.count_rows()} rows to "
        f"'{FRAMES_TABLE}' at {conn.describe()} in {time.perf_counter() - start:.0f}s"
    )
    print("Next: uv run python scripts/02_create_indexes.py")


if __name__ == "__main__":
    main()
