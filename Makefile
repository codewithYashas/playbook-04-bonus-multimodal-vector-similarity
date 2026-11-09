.PHONY: smoke embeddings analyze clean

# Sample data, embeddings on a subset, LanceDB search, the APIs, the quality tool, the dashboard. CPU only.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

# All 32 videos and 7 images (about 4-5 minutes on a laptop CPU), then the analysis
embeddings:
	uv run --frozen python src/embeddings/generator.py --input data/videos,data/images --output data/embeddings

analyze:
	uv run --frozen python analyze_embeddings.py
	uv run --frozen python test_similarity_search.py

# Remove generated embeddings, the LanceDB table and quality reports
clean:
	rm -rf data/embeddings data/quality data/lancedb
