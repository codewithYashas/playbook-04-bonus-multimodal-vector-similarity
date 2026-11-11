"""Smoke test: the pipeline on 3 videos + 7 images, the guide's numbers, and the app.

Run with ``make smoke`` (about a minute on a laptop CPU). The first run downloads the
ResNet-18 weights (about 45 MB). Kept despite the fe-demos "no tests" default because
the upstream repo had CI and the guide's numbers are worth pinning (see PLAN.md).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lancedb
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path[:0] = [str(ROOT), str(ROOT / "app")]

import queries as q  # noqa: E402

# The three videos the guide's quality-check examples use.
VIDEOS = ["download (25).mp4", "download (20).mp4", "download (15).mp4"]


def test_sample_data_complete_and_documented():
    readme = (DATA / "README.md").read_text()
    assert len(list((DATA / "videos").glob("*.mp4"))) == 32
    images = sorted(p.name for p in (DATA / "images").glob("*.jpg"))
    assert len(images) == 7
    assert all(name in readme for name in images)


def test_orders_name_no_real_restaurants():
    with (DATA / "orders" / "multirestaurant_orders.jsonl").open() as fh:
        names = {json.loads(line)["Restaurant name"] for line in fh}
    assert names == {"Sample Seafood House", "Sample Poke Bar", "Sample Bowl Kitchen"}


@pytest.fixture(scope="module")
def db_uri(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("byteeats")
    videos = tmp / "videos"
    videos.mkdir()
    for name in VIDEOS:
        shutil.copy(DATA / "videos" / name, videos / name)
    uri = str(tmp / "byteeats.lancedb")
    env = {**os.environ, "LANCEDB_PROFILE": ""}
    for script, extra in (("01_embed_media.py", ["--videos", str(videos)]), ("02_create_indexes.py", [])):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--uri", uri, *extra], check=True, env=env, cwd=ROOT
        )
    return uri


@pytest.fixture(scope="module")
def tables(db_uri):
    db = lancedb.connect(db_uri)
    return db.open_table("media"), db.open_table("frames")


def test_tables_hold_vectors_not_npy_files(db_uri, tables):
    media, frames = tables
    assert media.count_rows() == 10 and frames.count_rows() == 3 * 24
    assert str(media.schema.field("embedding").type) == "fixed_size_list<item: float>[512]"
    assert not list(Path(db_uri).parent.rglob("*.npy"))
    assert {i.index_type for i in frames.list_indices()} >= {"IvfHnswSq", "BTree"}


def test_image_to_video_search_matches_the_guide(tables):
    media, _ = tables
    tuna = q.media_row(media, "Tuna_Lover")
    found = q.media_search(media, tuna["embedding"], "Tuna_Lover", media_type="video", limit=1)
    assert found.iloc[0]["media_id"] == "download (15)"
    # The guide's analyze_embeddings.py output, with upstream's tolerance
    assert found.iloc[0]["similarity"] == pytest.approx(0.7764, abs=0.005)


@pytest.mark.parametrize(
    ("video", "dish", "threshold", "passed"),
    [
        ("download (25)", "Salmon_Edamame", 0.80, False),
        ("download (20)", "Salmon_Edamame", 0.75, False),
        ("download (15)", "Miso_Chicken", 0.75, True),
    ],
)
def test_quality_verdicts_match_the_guide(tables, video, dish, threshold, passed):
    media, frames = tables
    ref = q.media_row(media, dish)
    scored = q.frame_search(frames, ref["embedding"], dish, limit=100, ann=q.AnnOptions(exact=True), video_id=video)
    assert len(scored) == 24
    assert bool(scored["similarity"].max() >= threshold) == passed


def test_app_renders_and_traces_queries(db_uri, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BYTEEATS_URI", db_uri)
    monkeypatch.setenv("LANCEDB_PROFILE", "")
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    at.toggle(key="byteeats.debug").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(".search(" in c.value for c in at.code)
    details = [b for b in at.button if b.key and b.key.startswith("byteeats.details.")]
    details[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
