"""Compare vector index types on the frames table: recall against exact search, and latency.

The original playbook compared FAISS ``IndexFlatL2``, ``IndexIVFFlat`` and
``IndexHNSWFlat``. This runs the LanceDB equivalents, plus the quantised types, on a
scratch copy of ``frames`` so the app's table and index stay as they are:

- Flat: no index (``bypass_vector_index()``), the ground truth
- IVF_FLAT, IVF_PQ, IVF_SQ, IVF_HNSW_SQ

Every video and reference image vector in ``media`` is one query (39 queries), top 10.

    uv run python scripts/03_compare_indexes.py
    uv run python scripts/03_compare_indexes.py --nprobes 1   # see recall drop
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lancedb  # noqa: E402
from lancedb.index import IvfFlat, IvfHnswSq, IvfPq, IvfSq  # noqa: E402

from scripts.common import (  # noqa: E402
    DISTANCE_TYPE,
    FRAMES_TABLE,
    MEDIA_TABLE,
    add_connection_args,
    connect_db,
    resolve_connection,
)

K = 10


def configs(rows: int) -> dict[str, object]:
    parts = max(1, rows // 256)
    return {
        "IVF_FLAT": IvfFlat(distance_type=DISTANCE_TYPE, num_partitions=parts),
        "IVF_PQ": IvfPq(distance_type=DISTANCE_TYPE, num_partitions=parts, num_sub_vectors=32),
        "IVF_SQ": IvfSq(distance_type=DISTANCE_TYPE, num_partitions=parts),
        "IVF_HNSW_SQ": IvfHnswSq(distance_type=DISTANCE_TYPE, num_partitions=parts),
    }


def run(table, queries: np.ndarray, *, exact: bool, nprobes: int, refine: int | None) -> tuple[list[set], float]:
    found, start = [], time.perf_counter()
    for q in queries:
        s = table.search(q).distance_type(DISTANCE_TYPE).limit(K).select(["frame_id", "_distance"])
        if exact:
            s = s.bypass_vector_index()
        else:
            s = s.nprobes(nprobes)
            if refine:
                s = s.refine_factor(refine)
        found.append(set(s.to_pandas()["frame_id"]))
    return found, (time.perf_counter() - start) * 1000 / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nprobes", type=int, default=2)
    parser.add_argument("--refine-factor", type=int, default=None)
    add_connection_args(parser)
    args = parser.parse_args()

    conn = resolve_connection(args.uri, args.profile)
    if conn.is_remote:
        sys.exit("03_compare_indexes.py works on a local copy only; run it against the local database.")
    src = connect_db(conn)
    queries = np.stack(
        src.open_table(MEDIA_TABLE).search().select(["embedding"]).limit(10_000).to_pandas()["embedding"]
    )
    frames = src.open_table(FRAMES_TABLE).search().limit(100_000).to_arrow()

    scratch = Path(tempfile.mkdtemp(prefix="byteeats-index-"))
    try:
        table = lancedb.connect(scratch).create_table("frames", frames)
        truth, flat_ms = run(table, queries, exact=True, nprobes=args.nprobes, refine=None)
        print(
            f"{len(queries)} queries, top {K}, over {table.count_rows()} frames, nprobes={args.nprobes}, "
            f"refine_factor={args.refine_factor}\n"
        )
        print(f"{'Index':<12} {'build ms':>9} {'query ms':>9} {'recall@10':>10}")
        print(f"{'Flat':<12} {'-':>9} {flat_ms:>9.2f} {1.0:>10.3f}")
        for name, config in configs(table.count_rows()).items():
            start = time.perf_counter()
            table.create_index("embedding", config=config, replace=True)
            build_ms = (time.perf_counter() - start) * 1000
            got, ms = run(table, queries, exact=False, nprobes=args.nprobes, refine=args.refine_factor)
            recall = np.mean([len(g & t) / K for g, t in zip(got, truth, strict=True)])
            print(f"{name:<12} {build_ms:>9.0f} {ms:>9.2f} {recall:>10.3f}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
