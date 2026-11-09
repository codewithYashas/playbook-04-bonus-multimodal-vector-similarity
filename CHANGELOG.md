# Changelog

## 2026.11.1 (2026-11-09): LanceDB-native rebuild

Rebuilt in the fe-demos demo format. See README "Ported from the original".

### Changed
- Embeddings go straight into LanceDB: `scripts/01_embed_media.py` writes a `media` table (videos and reference images, with file bytes, thumbnails, metadata and vectors) and a `frames` table (every 5th frame's vector and thumbnail). No `.npy` or JSON sidecar files.
- `scripts/02_create_indexes.py` builds IVF_HNSW_SQ on `frames.embedding` and BTree/Bitmap scalar indexes; `scripts/03_compare_indexes.py` replaces the Flat/IVF/HNSW comparison with measured recall@10 for IVF_FLAT, IVF_PQ, IVF_SQ and IVF_HNSW_SQ.
- Cosine distance (similarity = 1 - distance) instead of squared L2 with 1 / (1 + d): the scores now match `analyze_embeddings.py`'s cosine similarities.
- One Streamlit app (`app/app.py`) replaces the FastAPI server, the quality API, the quality CLI and the Dash dashboard: Search, Quality check and Explore tabs, a details dialog with neighbours, and a Debug toggle showing raw rows and each LanceDB query.
- The quality check is a prefiltered vector search over the stored frames (24 per video) rather than re-decoding 8 frames per check. The guide's four examples keep their verdicts; their scores move by up to 0.03.
- Clustering: UMAP + K-Means/HDBSCAN on the 512-d vectors, named by the nearest reference dish, replacing the 25th-percentile-distance grouping and the PCA plot.

### Removed
- `src/` (FastAPI API, quality API, Dash dashboard, FAISS-era search and generator), `run_pipeline.sh`, `install.sh`, `analyze_embeddings.py`, the `test_*.py` API scripts, `scripts/{cleanup,quality_check,setup_directories}.py`, `data/sample_queries.json`, and `docs/` (the guide and its dashboard screenshots, which describe the removed UI; they remain upstream and in git history).
- Dependencies: fastapi, uvicorn, pydantic, python-multipart, requests, dash, tqdm. Added: streamlit, umap-learn, pyarrow; ruff (dev).

## 2026.11 (2026-11-09): LanceDB port

### Changed
- Similarity search uses LanceDB (embedded, local) instead of FAISS. `load_embeddings()` writes the
  embeddings and their metadata to a Lance table (`data/lancedb/`, or `LANCEDB_URI`); `build_index()`
  maps Flat to exact search, IVF to `IVF_FLAT` and HNSW to `IVF_HNSW_FLAT`. Squared L2 distance and
  the 1 / (1 + distance) score are unchanged, so the guide's scores are too.
- Installs use uv only. `install.sh` runs `uv sync --frozen`, `run_pipeline.sh` runs Python through
  `uv run`, and `requirements.txt` and the pip CI step are removed.
- `faiss-cpu` replaced by `lancedb` in `pyproject.toml` and `uv.lock`.

### Removed
- The macOS OpenMP troubleshooting note: it came from FAISS and PyTorch each bundling OpenMP.

## 2026.10.1 (2026-10-27)

### Changed
- README: this repository is bonus content; the book doesn't print it (checked against the Early
  Release). The differences section compares with the first edition (the `first-edition` tag).

## 2026.10 (2026-10-23): 2026 maintenance update

The first-edition version is tagged `first-edition`.

### Fixed
- `install.sh` rejected every Python from 3.10 on: its check read "3.12" as 31 and compared it with 38. It now asks the interpreter and requires 3.12+; `PYTHON=` picks the interpreter.
- The API read `EMBEDDING_DIR` while `run_pipeline.sh` exported `EMBEDDINGS_DIR`, so a custom embeddings folder was ignored. Both names work.
- The API loaded its embeddings and models twice at startup (an explicit call plus the lifespan hook).
- `test_similarity_search.py`'s custom query used unseeded random noise, so its scores changed on every run. It is seeded now.
- `scripts/cleanup.py --temp` and `--pycache` also deleted files inside `venv/`. They skip `.git` and virtual environments.
- Guide step 5 showed the API command with the dashboard's output; it now shows both commands.
- README: the real clone URL and folder, Python 3.12+.

### Security
- The dashboard ran Flask/Dash debug mode on `0.0.0.0`, and both APIs listened on `0.0.0.0`. All three listen on `127.0.0.1` now with debug off. `HOST=0.0.0.0` and `DASH_DEBUG=1` opt back in.

### Changed
- `data/orders/multirestaurant_orders.jsonl`: the invented orders, ratings and reviews named three real San Francisco restaurants. They name fictional ones now (Sample Seafood House, Sample Poke Bar, Sample Bowl Kitchen); nothing else changed.
- The quality checker loads ResNet with `weights=ResNet18_Weights.IMAGENET1K_V1` instead of the deprecated `pretrained=True` (the same weights).
- Dependencies: pinned in `requirements.txt`, exported from `uv.lock`; Python 3.12+, NumPy 2, OpenCV 5 (headless build; no window functions are used), PyTorch's CPU build on Linux. The embeddings and analysis output are identical to the first-edition versions. matplotlib and pyyaml are no longer listed (nothing imports them); requests is listed (the quality tool imports it).
- README in the standard layout, with Tested on, Differences, Data and licenses, Expected output from a fresh-clone run, and a macOS OpenMP note.

### Added
- `data/README.md`: the source (created by the author with a Google generative video model; orders generated by the author) and MIT license of the sample data, and what uses each file.
- `pyproject.toml`, `uv.lock`, `tests/test_smoke.py`, `Makefile`, and GitHub Actions on Python 3.12 and 3.13, including the printed `pip install -r requirements.txt` path.

### Removed
- Outputs of April 2025 runs from git: `data/embeddings/` (39 metadata files without their embeddings, and batch summaries), `data/embeddings_test/`, `data/results/`. The steps regenerate them, and they are ignored now.
