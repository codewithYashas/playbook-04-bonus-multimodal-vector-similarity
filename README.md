# ByteEats Vector Similarity Search

**Bonus content for Chapter 4** of *Data Engineering for Multimodal AI* (O'Reilly, 2026). It is not printed in the book; it extends the [Chapter 4 playbook](https://github.com/Data-Engineering-for-Multimodal-AI/playbook-04-multimodal-ingestion-pipeline).
Book: https://oreil.ly/data-engineering-multimodal

> **LanceDB port.** This copy replaces the in-memory FAISS index with [LanceDB](https://lancedb.com), an embedded vector database: the embeddings and their metadata are written to a Lance table on local disk (`data/lancedb/`), and the Flat, IVF and HNSW searches run against it. There is no server to run. Installs use [uv](https://docs.astral.sh/uv/) only. See [Differences from the original (LanceDB port)](#differences-from-the-original-lancedb-port).

> Want the code as it was when the book was published? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the first edition](#differences-from-the-first-edition).

## What you'll build

An advanced vector embedding and similarity search system for food preparation videos. This system demonstrates complex data transformation by converting unstructured video data into structured vector embeddings and implementing efficient similarity search.

- **Video Feature Extraction**: Extracts meaningful features from food preparation videos
- **Vector Embedding Generation**: Transforms video features into 512-dimensional vector embeddings using ResNet
- **Similarity Search**: Efficiently finds similar food preparations using vector similarity with LanceDB (embedded)
- **Quality Assessment**: Compares videos against reference images with configurable thresholds
- **Visualization**: Interactive dashboard to explore embeddings and quality assessment results

Everything runs on a laptop CPU with the bundled sample data (32 videos, 7 reference images). The first run downloads the ResNet-18 weights (about 45 MB) from download.pytorch.org.

For a step-by-step walkthrough with expected output and dashboard screenshots, see the **[Bonus Content Guide](docs/GUIDE.md)**. Its results reproduce exactly with this version (see [Expected output](#expected-output)).

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and about 1.4 GB of disk space for the environment. uv downloads Python 3.12 or newer if your system doesn't have it.

```bash
# Clone the repository
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-04-bonus-multimodal-vector-similarity.git
cd playbook-04-bonus-multimodal-vector-similarity

# Install the locked dependencies into .venv
./install.sh        # or: uv sync
```

The versions are pinned in `uv.lock`; on Linux, PyTorch comes from its CPU-only index (see `pyproject.toml`). There is no `requirements.txt` and no virtual environment to activate: prefix Python commands with `uv run`. `run_pipeline.sh` and the `Makefile` already do.

Then:

```bash
export IMAGE_DIR=data/images
./run_pipeline.sh --generate-embeddings     # about 3-5 minutes on a laptop CPU
uv run python analyze_embeddings.py
uv run python test_similarity_search.py     # or: make analyze
```

and the API, quality check tool and dashboard as described below. The servers listen on this machine only (`127.0.0.1`); set `HOST=0.0.0.0` to reach them from another machine.

### Checks

```bash
make smoke    # or: uv run pytest -q
```

`make smoke` (about a minute on a laptop CPU) generates embeddings for 3 videos and the 7 reference images, then checks LanceDB search (Flat and HNSW), the analysis scripts, the dashboard's plot, the API through `run_pipeline.sh --start-api`, the quality check tool, and three scores from the guide. GitHub Actions runs it on Python 3.12 and 3.13.

## Detailed Implementation

### 1. Embedding Generation

The embedding generation process converts videos and images into 512-dimensional vectors using a pre-trained ResNet model:

```bash
export IMAGE_DIR=data/images
./run_pipeline.sh --generate-embeddings
```

#### How it Works:

1. **Video Processing**:
   - Loads videos from the specified directory
   - Extracts frames at regular intervals (every 5th frame by default)
   - Processes each frame through ResNet to extract features
   - Averages the frame embeddings to create a single video embedding
   - Normalizes the embeddings to ensure consistent similarity calculations

2. **Image Processing**:
   - Loads reference images from the images directory
   - Processes each image through the same ResNet model
   - Normalizes the embeddings for consistent comparison with video embeddings

3. **Output**:
   - Each embedding is saved as a .npy file in the data/embeddings directory
   - Metadata is saved alongside each embedding in JSON format
   - For videos: includes source path, duration, frame count, and processing time
   - For images: includes source path, resolution, and processing time

#### Technical Details:

- Uses transfer learning with ResNet-18 pretrained on ImageNet
- Removes the classification layer to extract 512-dimensional feature vectors
- Applies L2 normalization to embeddings for optimal cosine similarity calculation
- Processes both images and videos through the same pipeline for compatibility

### 2. Similarity Search API

The API server provides endpoints for searching similar videos and performing quality checks:

```bash
export API_PORT=8085
export IMAGE_DIR=data/images
./run_pipeline.sh --start-api
```

#### API Architecture:

1. **FastAPI Backend**:
   - RESTful API with documented endpoints (Swagger available at /docs)
   - Handles requests for similarity search and quality assessment
   - Writes the embeddings to a local LanceDB table at startup and searches it
   - Processes quality check requests by comparing videos to reference images

2. **LanceDB Table and Index**:
   - Stores each embedding with its ID and metadata in a Lance table under `data/lancedb/` (set `LANCEDB_URI` to use another location)
   - Supports different index types: Flat (exact search, no index), IVF (`IVF_FLAT`) and HNSW (`IVF_HNSW_FLAT`)
   - Uses squared L2 distance, like the original FAISS `IndexFlatL2`, so the similarity scores are unchanged
   - Rewritten from the embeddings folder each time the API starts

3. **Quality Check**:
   - Loads video frames and reference images
   - Computes frame-by-frame similarity scores
   - Determines quality based on configurable thresholds
   - Generates detailed reports saved to the quality directory

#### API Endpoints:

- `GET /api/videos`: List all available videos
- `GET /api/quality/reference_images`: List all reference images
- `POST /api/quality/check`: Check quality of a video against a reference image
- `GET /api/quality/reports`: Get all quality reports
- `GET /api/similar/{video_id}`: Find videos similar to a specific video
- `POST /api/search`: Search for videos using a custom query

### 3. Dashboard Visualization

The dashboard provides an interactive interface to explore embeddings and assess quality:

```bash
export API_PORT=8085
export DASHBOARD_PORT=8095
export API_URL="http://localhost:8085"
./run_pipeline.sh --start-dashboard
```

#### Dashboard Features:

1. **Vector Embedding Visualization**:
   - Interactive 2D plot using PCA (Principal Component Analysis)
   - Visualizes relationships between videos and reference images
   - Color-coded data points for easy identification
   - Hover tooltips with detailed information

2. **Quality Assessment Interface**:
   - Select videos and reference images from dropdown menus
   - Configure quality threshold with a slider (0.5-0.95)
   - View side-by-side comparison of video frames and reference images
   - See detailed similarity scores and assessment results

3. **Similarity Search**:
   - Find videos similar to a selected video
   - View ranked results with similarity scores
   - Visualize embedding clusters and patterns

#### Technical Implementation:

- Built with Dash and Plotly for interactive visualizations
- Communicates with the API server using HTTP requests
- Updates visualizations in real-time as new data becomes available
- Responsive design for different screen sizes

### 4. Quality Check Tool

The quality check tool provides a command-line interface for assessing video quality:

```bash
export API_PORT=8085
export IMAGE_DIR=data/images
./run_pipeline.sh --quality-check
```

#### Quality Assessment Process:

1. **Initialization**:
   - Loads available videos and reference images
   - Prompts user to select a video, reference image, and threshold

2. **Assessment**:
   - Extracts frames from the selected video
   - Generates embeddings for video frames
   - Compares each frame to the reference image embedding
   - Finds the best matching frame and its similarity score

3. **Result Determination**:
   - PASSED: Similarity score ≥ threshold (good quality match)
   - FAILED: Similarity score < threshold (poor quality match)
   - Quality reports include detailed frame-by-frame similarity metrics

4. **Output**:
   - Detailed assessment with similarity scores and threshold information
   - Processing time and technical details
   - Option to view all quality reports in a tabular format

#### Similarity Score Interpretation:

- 0.75-0.85: Excellent match (very similar preparation)
- 0.65-0.75: Good match (similar preparation with minor differences)
- 0.55-0.65: Fair match (recognizable but with significant differences)
- Below 0.55: Poor match (different preparation or significant issues)

## Troubleshooting

### Common Issues:

1. **Port Conflicts**:
   - If you encounter "address already in use" errors, change the port:
   ```bash
   export API_PORT=8086  # Use a different port number
   ```

2. **Image Directory Not Found**:
   - Ensure the IMAGE_DIR environment variable points to the correct path:
   ```bash
   export IMAGE_DIR=data/images
   ```

3. **No Embeddings Found**:
   - Run the embedding generation step first:
   ```bash
   ./run_pipeline.sh --generate-embeddings
   ```

4. **Dashboard Connection Issues**:
   - Make sure the API server is running
   - Verify the API_URL points to the correct address and port:
   ```bash
   export API_URL="http://localhost:8085"
   ```

5. **`No existing dataset at .../embeddings.lance, it will be created`**: a harmless LanceDB message the first time the table is written.

6. **Quality tool numbering**: the tool numbers videos in the order the file system lists them, which can differ between machines. Check the name it prints ("Checking quality of download (25) ...") rather than relying on the number.

7. **`[h264 @ ...] mmco: unref short failure`** while generating embeddings: a harmless message from the video decoder about a few of the clips. The frames are still read.

8. **`install.sh` says uv is not installed**: install it from https://docs.astral.sh/uv/getting-started/installation/ (for example `brew install uv`).

## Tested on

| | |
|---|---|
| Date | 2026-10-23 |
| OS | macOS 26.5.1 (arm64); GitHub Actions `ubuntu-latest` |
| Python | 3.12.15 and 3.13.16 |
| Key packages (original) | torch 2.14.1, torchvision 0.29.1, faiss-cpu 1.15.1, numpy 2.5.4, opencv-python-headless 5.0.0.93, fastapi 0.162.0, dash 4.4.1, plotly 7.1.0, pandas 3.1.0 |
| Install time | `./install.sh`: 68 s (environment 1.2 GB) |
| Embeddings | All 32 videos and 7 images: about 3 minutes (170 s) on an Apple-silicon laptop CPU |

LanceDB port: `make smoke` passes on 2026-11-09, macOS 26.7.1 (arm64), Python 3.14.8, lancedb 0.41.0, pyarrow 26.0.1. The other rows are from the original FAISS version.

Every command in this README and the guide was run from a fresh clone of this branch on Python 3.12. The embeddings give the same analysis output as the first-edition code with its original versions (NumPy 1.26, OpenCV 4.11), and as the guide.

## Differences from the first edition

| What | First edition | Now | Why |
|---|---|---|---|
| Clone URL and folder | `github.com/yourusername/ByteEatsVectorSimilaritySearch` | `playbook-04-bonus-multimodal-vector-similarity` | The repository's actual location |
| Python | 3.8+ (`install.sh`), 3.10+ (README) | 3.12+ | Current dependencies |
| `install.sh` | Rejected every Python from 3.10 on ("Python 3.8 or higher is required") | Accepts 3.12+; `PYTHON=` picks the interpreter | Its version check read 3.12 as 31 |
| Servers | API and quality API on `0.0.0.0`; dashboard on `0.0.0.0` in Flask/Dash debug mode | `127.0.0.1`, debug off; `HOST=0.0.0.0` and `DASH_DEBUG=1` opt in | Anyone on the network could reach them, and debug mode must not be exposed |
| Dashboard startup line | `Dash is running on http://0.0.0.0:8095/` | `Dash is running on http://127.0.0.1:8095/` | Same reason |
| Guide step 5 | Showed the API command (`--start-api`) with the dashboard's output | Shows both commands | Typo |
| `test_similarity_search.py` custom query | Random noise, so its scores changed on every run | Seeded: the same scores every run | Reproducible output |
| API embeddings folder | `run_pipeline.sh` exported `EMBEDDINGS_DIR`; the API read `EMBEDDING_DIR` | Both work | Bug |
| API startup | Loaded the embeddings and models twice | Once | Left over from the `on_event` to `lifespan` migration |
| Quality checker model | `resnet18(pretrained=True)` (deprecation warning) | `weights=ResNet18_Weights.IMAGENET1K_V1` | Same weights; `pretrained=` is deprecated |
| `scripts/cleanup.py` | `--temp` and `--pycache` also cleaned inside `venv/` | Skips `.git` and virtual environments | Bug |
| Committed outputs | `data/embeddings/` (39 metadata files without their embeddings), `data/embeddings_test/`, `data/results/` from April 2025 runs | Not in git; generated by the steps | Every run rewrote them |
| Sample orders | Named three real San Francisco restaurants, with invented ratings and reviews | Fictional names (Sample Seafood House, Sample Poke Bar, Sample Bowl Kitchen) | The reviews were invented; they shouldn't name real businesses |
| Dependencies | Version ranges, `numpy<2`, no lock | Pinned from `uv.lock`; NumPy 2, OpenCV 5 (headless); PyTorch CPU build on Linux | Reproducible installs; the results are unchanged |

Details are in [CHANGELOG.md](CHANGELOG.md).

## Differences from the original (LanceDB port)

| What | Original | LanceDB port |
|---|---|---|
| Vector store | In-memory FAISS index, rebuilt on every start | LanceDB table on local disk (`data/lancedb/`, or `LANCEDB_URI`), rewritten from the embeddings folder on every start |
| Index types | `IndexFlatL2`, `IndexIVFFlat`, `IndexHNSWFlat` | No index (exact search), `IVF_FLAT`, `IVF_HNSW_FLAT` |
| Distance and scores | Squared L2; score = 1 / (1 + distance) | The same, so the guide's scores are unchanged |
| Metadata | Python lists next to the index | A JSON column in the same table as the vectors |
| Install | `install.sh` (venv + pip), `requirements.txt`, or `uv sync` | uv only: `install.sh` runs `uv sync --frozen`; `run_pipeline.sh` runs Python through `uv run` |
| macOS OpenMP conflict | FAISS and PyTorch each bundle OpenMP, so one process using both could abort or hang | LanceDB doesn't link OpenMP; a PyTorch model and a LanceDB search run in one process |

## Data and licenses

| Files | Source | License |
|---|---|---|
| `data/videos/` (32 clips), `data/images/` (7 reference photos) | Created by the author with a Google generative video model | MIT |
| `data/orders/multirestaurant_orders.jsonl` | Generated by the author; fictional restaurants | MIT |
| `docs/images/` (11 screenshots and frames) | The author's runs of the dashboard | MIT |

None of it shows a real restaurant, kitchen, robot, customer or person. [data/README.md](data/README.md) lists every file and what uses it.

## Expected output

From a fresh clone on 2026-10-23, macOS 26.5.1 (arm64), Python 3.12.15.

**Embedding generation** (`./run_pipeline.sh --generate-embeddings`; progress bars and decoder messages removed):

```
Using device: cpu

Processing directory: data/videos
In directory data/videos:
  Found 32 video files and 0 image files
Processing 32 videos...

Processing directory: data/images
In directory data/images:
  Found 0 video files and 7 image files
Processing 7 images...

Summary:
Processed 32 videos and 7 images
Total processing time: 122.65 seconds
Average processing time: 3.145 seconds per file
Embeddings saved to data/embeddings
Embeddings generation complete.
```

**`python analyze_embeddings.py`**: identical to [Step 2 of the guide](docs/GUIDE.md#step-2-analyzing-the-embeddings), for example:

```
Found 32 video embeddings and 7 image embeddings
Video embedding shape: (512,)
Image embedding shape: (512,)
✓ Embeddings have compatible shapes

Top video matches for each reference image:

Tuna_Lover:
  1. download (15): 0.7764
  2. download (12): 0.7535
  3. download (14): 0.7260
...
Embedding statistics:
Average video embedding magnitude: 0.8971
Average image embedding magnitude: 0.9361
Similarity between average video and image embedding: 0.7708
```

**`python test_similarity_search.py`**: identical to [the guide](docs/GUIDE.md#running-the-test-script) up to the custom query, which is now seeded:

```
Custom query using Tuna_Lover embedding:
  1. Tuna_Lover: 0.6268
  2. Miso_Salmon: 0.5974
  3. House_Salmon: 0.5850
  4. Miso_Chicken: 0.5688
  5. Kid's_Crab_Bowl: 0.5591
```

**Quality checks**, through the API and the quality tool: the guide's four examples give the same scores.

| Video | Reference image | Threshold | Score | Result |
|---|---|---|---|---|
| download (25) | Salmon_Edamame.jpg | 0.80 | 0.7753 | FAILED (Needs review) |
| download (20) | Salmon_Edamame.jpg | 0.75 | 0.6371 | FAILED (Needs review) |
| download (15) | Miso_Chicken.jpg | 0.75 | 0.7522 | PASSED (Acceptable) |
| download (9) | Miso_Salmon.jpg | 0.65 | 0.6640 | PASSED (Acceptable) |

**API startup** (`./run_pipeline.sh --start-api` with `API_PORT=8085`; ready after about 30 seconds):

```
Starting ByteEats Vector Similarity Search API on port 8085
Loading 39 embeddings...
Loaded 39 embeddings with shape (39, 512)
Built Flat index with 39 vectors in 0.003 seconds
Using device: cpu
Using device: cpu
API initialized with 39 embeddings
INFO:     Uvicorn running on http://127.0.0.1:8085 (Press CTRL+C to quit)
```

`test_api_client.py` and `test_quality_api.py` run against it and end with `All API tests completed!`. The dashboard (`./run_pipeline.sh --start-dashboard`) prints `Dash is running on http://127.0.0.1:8095/`, and its PCA plot is titled `PCA Projection (Explained variance: 0.28, 0.07)`.

## Project structure

```
├── src/
│   ├── embeddings/generator.py       # ResNet-18 video and image embeddings
│   ├── similarity/search.py          # LanceDB table, index (Flat, IVF, HNSW) and search
│   ├── api/main.py                   # Similarity search API (includes the quality endpoints)
│   ├── api/quality_check_api.py      # Quality check endpoints
│   ├── api/run_quality_api.py        # Quality check API on its own
│   ├── api/dashboard.py              # Dash dashboard
│   └── utils/                        # quality_checker.py, init.py (setup helpers)
├── scripts/                          # quality_check.py (CLI tool), setup_directories.py, cleanup.py
├── data/                             # Sample videos, images and orders (see data/README.md); data/lancedb/ is generated
├── docs/GUIDE.md, docs/images/       # Bonus content guide and screenshots
├── analyze_embeddings.py, test_similarity_search.py, test_api_client.py, test_quality_api.py
├── run_pipeline.sh, install.sh       # Both run through uv
├── tests/test_smoke.py, Makefile
├── pyproject.toml, uv.lock           # Dependencies (uv)
├── CHANGELOG.md
└── LICENSE
```

## Performance Considerations

For optimal performance when scaling to larger datasets:

1. **Embedding Generation**:
   - Processing thousands of videos requires distributed computation
   - Consider using GPU acceleration for 10-20x speedup
   - Batch processing for efficient resource utilization

2. **Similarity Search**:
   - LanceDB indexes can be configured for different performance profiles:
     - Flat: Most accurate but slower for large collections
     - IVF (`IVF_FLAT`): Faster but slightly less accurate
     - HNSW (`IVF_HNSW_FLAT`): Best balance of speed and accuracy for large datasets
     - Quantized variants (`IVF_PQ`, `IVF_HNSW_SQ`, `IVF_RQ`) trade some accuracy for smaller indexes

3. **Memory Management**:
   - The Lance table is read from disk, so the embeddings don't need to fit in memory
   - Implement pagination for API results
   - Use quantization techniques to reduce embedding size (FP16, INT8)

## Data Engineering Highlights

This project showcases advanced data engineering techniques:

- **Complex Data Transformation**: Converting raw video data into structured vector embeddings
- **Transfer Learning**: Using pre-trained ResNet models to extract visual features
- **Temporal Aggregation**: Averaging frame embeddings for consistent video representation
- **Fast Similarity Search**: Using LanceDB, an embedded vector database, for efficient nearest-neighbor queries
- **Cross-Modal Comparison**: Matching videos against images using the same embedding space

## License

The ByteEats Vector Similarity Search project is licensed under the MIT License. The sample data in `data/` and the screenshots in `docs/images/` are MIT-licensed too.
