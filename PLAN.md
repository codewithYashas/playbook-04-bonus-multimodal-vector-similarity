# PLAN: ByteEats vector similarity on LanceDB

## Problem

The book's Chapter 4 bonus playbook shows video and image similarity search for a ghost kitchen: ResNet-18 embeddings of 32 prep videos and 7 reference dish photos, nearest-neighbour search with FAISS (flat, IVF, HNSW), a frame-by-frame quality check against a reference photo, and a Dash dashboard.

It's a natural FAISS → LanceDB migration story (FENG-1890, playbook #7) and a Stage 3 "load data, build vectors, run queries" prototype. A first pass swapped FAISS for a LanceDB table but kept the rest: embeddings still went to `.npy` files, the table was rebuilt at every API start, the quality check re-decoded videos, and neighbours and clusters were JSON from an API. This pass rebuilds it in the fe-demos [demo format](https://github.com/lancedb/fe-demos/blob/main/demo_format.md), following the enterprise demo's Vector Search page.

## Data sources

The repo's own sample data (MIT, by the book's author): 32 five-second MP4 clips, 7 JPEG reference photos, 500 invented orders. Nothing to download except the ResNet-18 weights.

## Schema

Two tables, defined with explicit Arrow schemas in `scripts/common.py` (see the README for every column):

- `media`: a row per video and reference image, holding the file bytes, a thumbnail, metadata and the 512-d vector. 39 rows.
- `frames`: a row per sampled frame (every 5th, as the original generator sampled), holding a thumbnail and the vector. 768 rows.

The video vector is the normalised mean of its frames' raw features, exactly as in the original, so the book's numbers reproduce.

## Approach

- `scripts/01_embed_media.py` embeds and writes both tables; `02_create_indexes.py` adds IVF_HNSW_SQ on `frames.embedding` and scalar indexes for the filters; `03_compare_indexes.py` keeps the book's index comparison as a measured recall/latency table.
- The app only opens tables and searches. Every read is a `Step` list that runs and prints identically, for the Debug panels.
- Cosine distance, so scores read as similarities and match the book's `analyze_embeddings.py`.
- The quality check becomes a prefiltered vector search over stored frames, instead of decoding the video and re-running the model per check.
- Neighbours and clusters move from JSON endpoints and a PCA scatter to a details dialog (ranked bars, thumbnails, a clickable map) and an Explore tab (UMAP + K-Means/HDBSCAN, clusters named by the nearest dish via a centroid search).

## Deviations from the demo format

- **Single-demo repo, kept as a fork.** The layout is the format's `app/` + `scripts/`, but the repo stays a fork of the book's repo so the port is a reviewable diff. `LICENSE` (MIT, the author's) stays, as the license requires.
- **Store name.** `byteeats.lancedb/`, after the playbook's fictional company, rather than the long repo name.
- **Data in git.** `data/` holds the book's sample media and stays committed (54 MB); only `byteeats.lancedb/` is generated. `.gitignore` excludes the store, not `data/`.
- **A smoke test and CI.** The upstream repo had them, and they pin the book's numbers (0.7764 for Tuna Lover → download (15), and the four quality verdicts) so a refactor can't silently change the results.
- **No LLM.** Clusters are named by the nearest reference dish, a LanceDB query, so there is no Anthropic dependency to degrade.
- **No FTS or hybrid search.** The data has no text worth indexing; dish names are seven words. See TODO.md.

## Open questions

- Should the port swap ResNet-18 for CLIP? It would add text-to-video search and likely separate the dishes better, but the book's numbers would no longer reproduce.
- Is the orders table worth loading to show structured and media data side by side, given there's no key linking orders to videos?
