"""Smoke tests: sample data, embeddings, LanceDB search, the APIs, the quality tool and the dashboard.

Run with `make smoke` (or `uv run pytest -q`). CPU only, a few minutes.
To stay short, the pipeline runs on 3 of the 32 videos plus the 7 reference images.
The first run downloads the ResNet-18 weights (about 45 MB) from download.pytorch.org.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

VIDEOS = ["download (25).mp4", "download (20).mp4", "download (15).mp4"]
IMAGES = sorted(os.listdir(os.path.join(DATA, "images")))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ data

def test_sample_data_complete_and_documented():
    import cv2
    with open(os.path.join(DATA, "README.md")) as fh:
        readme = fh.read()
    videos = sorted(os.listdir(os.path.join(DATA, "videos")))
    assert len(videos) == 32 and len(IMAGES) == 7
    for name in videos[:3]:
        cap = cv2.VideoCapture(os.path.join(DATA, "videos", name))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        cap.release()
    for name in IMAGES:
        assert name in readme
        assert cv2.imread(os.path.join(DATA, "images", name)) is not None


def test_orders_name_no_real_restaurants():
    with open(os.path.join(DATA, "orders", "multirestaurant_orders.jsonl")) as fh:
        names = {json.loads(line)["Restaurant name"] for line in fh}
    assert names == {"Sample Seafood House", "Sample Poke Bar", "Sample Bowl Kitchen"}


# ------------------------------------------------------------------ pipeline on a subset

@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """A copy of the repository layout with 3 videos and 7 images, embeddings generated."""
    work = tmp_path_factory.mktemp("bonus")
    (work / "data" / "videos").mkdir(parents=True)
    for name in VIDEOS:
        shutil.copy(os.path.join(DATA, "videos", name), work / "data" / "videos" / name)
    shutil.copytree(os.path.join(DATA, "images"), work / "data" / "images")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "src", "embeddings", "generator.py"),
                             "--input", "data/videos,data/images", "--output", "data/embeddings"],
                            cwd=work, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Processed 3 videos and 7 images" in result.stdout
    return work


def test_embeddings(workdir):
    files = sorted(f for f in os.listdir(workdir / "data" / "embeddings") if f.endswith("_embedding.npy"))
    assert len(files) == 10
    for f in files:
        vector = np.load(workdir / "data" / "embeddings" / f)
        assert vector.shape == (512,)
    with open(workdir / "data" / "embeddings" / "download (25)_metadata.json") as fh:
        assert json.load(fh)["model_used"] == "resnet18"


def py(code, cwd, env=None):
    """Run Python in a fresh process, the way the project's scripts and servers run."""
    result = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
                            timeout=600, env=dict(os.environ, PYTHONPATH=ROOT, **(env or {})))
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_similarity_search(workdir):
    out = py("""
import json
from src.similarity.search import SimilaritySearch
search = SimilaritySearch("data/embeddings")
print(search.load_embeddings())
for index_type in ("Flat", "HNSW"):
    search.build_index(index_type=index_type)
    results, _ = search.find_similar_by_id("Tuna_Lover", k=3)
    print(json.dumps([[r["video_id"], round(float(r["similarity_score"]), 2)] for r in results]))
""", cwd=workdir)
    lines = out.strip().splitlines()
    assert "10" in lines
    flat, hnsw = [json.loads(line) for line in lines if line.startswith("[[")]
    # Image-to-image similarity from docs/GUIDE.md (Step 3): Miso_Salmon 0.8830, House_Salmon 0.8573
    assert flat == [["Tuna_Lover", 1.0], ["Miso_Salmon", 0.88], ["House_Salmon", 0.86]]
    assert hnsw[0] == ["Tuna_Lover", 1.0]


def test_analyze_and_search_scripts(workdir):
    results = {}
    for script in ("analyze_embeddings.py", "test_similarity_search.py"):
        result = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=workdir,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        results[script] = result.stdout
    assert "Found 3 video embeddings and 7 image embeddings" in results["analyze_embeddings.py"]
    assert "All tests completed successfully!" in results["test_similarity_search.py"]


def test_dashboard_plot(workdir):
    out = py("""
import src.api.dashboard as dashboard
dashboard.parent_dir = "."
figure = dashboard.update_embedding_plot("pca")
print(figure.layout.title.text)
print(len(figure.data[0].x))
""", cwd=workdir)
    title, points = out.strip().splitlines()[-2:]
    assert title.startswith("PCA Projection") and points == "10"


# ------------------------------------------------------------------ servers and the quality tool

def test_api_server_and_quality_tool(workdir):
    """run_pipeline.sh --start-api, then the quality tool's non-interactive mode, as a reader would."""
    port = free_port()
    env = dict(os.environ, API_PORT=str(port), IMAGE_DIR="data/images", VIDEO_DIR="data/videos",
               EMBEDDINGS_DIR="data/embeddings", QUALITY_DIR="data/quality")
    shutil.copy(os.path.join(ROOT, "run_pipeline.sh"), workdir / "run_pipeline.sh")
    for folder in ("src", "scripts"):
        if not (workdir / folder).exists():
            shutil.copytree(os.path.join(ROOT, folder), workdir / folder,
                            ignore=shutil.ignore_patterns("__pycache__"))
    server = subprocess.Popen(["bash", "run_pipeline.sh", "--start-api"], cwd=workdir, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(240):
            try:
                urllib.request.urlopen(url + "/", timeout=1)
                break
            except OSError:
                assert server.poll() is None, "API exited during startup"
                time.sleep(0.5)
        videos = json.load(urllib.request.urlopen(url + "/api/videos"))
        assert {v[:-4] for v in VIDEOS} <= {v["video_id"] for v in videos}   # lists the images too
        similar = json.load(urllib.request.urlopen(url + "/api/similar/download%20%2825%29"))["results"]
        assert similar[0]["video_id"] == "download (25)"
        request = urllib.request.Request(url + "/api/quality/check", method="POST",
                                         headers={"Content-Type": "application/json"},
                                         data=json.dumps({"video_id": "download (25)",
                                                          "reference_image": "Salmon_Edamame.jpg",
                                                          "threshold": 0.80}).encode())
        report = json.load(urllib.request.urlopen(request, timeout=120))
        # docs/GUIDE.md: download (25) vs Salmon_Edamame.jpg scores 0.7753 and fails at 0.80
        assert report["quality_passed"] is False
        assert report["max_similarity_score"] == pytest.approx(0.7753, abs=0.005)
        result = subprocess.run([sys.executable, "scripts/quality_check.py", "--api-url", url,
                                 "--video", "download (20)", "--reference", "Salmon_Edamame.jpg",
                                 "--threshold", "0.75"], cwd=workdir, env=env,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        # docs/GUIDE.md: download (20) vs Salmon_Edamame.jpg scores 0.6371 and fails at 0.75
        assert "QUALITY CHECK: FAILED" in result.stdout
        assert "Similarity Score: 0.63" in result.stdout or "Similarity Score: 0.64" in result.stdout
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_servers_listen_locally_by_default():
    for path in ("src/api/main.py", "src/api/run_quality_api.py", "src/api/dashboard.py"):
        with open(os.path.join(ROOT, path)) as fh:
            source = fh.read()
        assert 'os.environ.get("HOST", "127.0.0.1")' in source, path
        assert 'host="0.0.0.0"' not in source, path


def test_install_uses_uv_and_python_312():
    # Installs go through uv and the lockfile; uv enforces requires-python (3.12+)
    with open(os.path.join(ROOT, "install.sh")) as fh:
        assert "uv sync --frozen" in fh.read()
    with open(os.path.join(ROOT, "run_pipeline.sh")) as fh:
        assert "\n    python " not in fh.read()
    with open(os.path.join(ROOT, "pyproject.toml")) as fh:
        assert 'requires-python = ">=3.12"' in fh.read()
