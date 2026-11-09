.PHONY: data embed index compare run smoke check clean

# Embed the 32 videos and 7 reference images into byteeats.lancedb/, then index (about 30 s on a laptop CPU)
data: embed index

embed:
	uv run python scripts/01_embed_media.py

index:
	uv run python scripts/02_create_indexes.py

# Flat vs IVF_FLAT / IVF_PQ / IVF_SQ / IVF_HNSW_SQ on the frames table: recall@10 and latency
compare:
	uv run python scripts/03_compare_indexes.py

run:
	uv run streamlit run app/app.py

# Pipeline on 3 videos + 7 images, the guide's numbers, and the app (headless). About a minute.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

check:
	uv run ruff check .
	uv run ruff format --check .

clean:
	rm -rf byteeats.lancedb
