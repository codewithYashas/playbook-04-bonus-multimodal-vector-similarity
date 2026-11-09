# TODO

- Test the LanceDB Cloud / Enterprise path end to end: `01_embed_media.py --profile ...` (chunked writes), `02_create_indexes.py` (background index builds), and the page against `db://`.
- CLIP embeddings as a second vector column (`clip_embedding`), for text-to-video search ("salmon bowl with edamame") and hybrid search over a caption column. Keep ResNet-18 so the book's numbers still reproduce.
- Load `data/orders/multirestaurant_orders.jsonl` as an `orders` table, and decide how (or whether) orders link to dishes and videos.
- Check tab: let the presenter add a reference photo by upload (embedded in the page; not written back, since the page is read-only).
- Explore: a 3-D projection option, as in the enterprise Vector Search page.
- `03_compare_indexes.py`: a synthetic scale-up option (for example 100k jittered frame vectors) so latency differences between index types show, not just recall.
