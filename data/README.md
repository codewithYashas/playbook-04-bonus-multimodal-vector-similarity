# Sample data

The book's author, Vasundra Srinivasan, created the videos and images here with a Google generative video model, and generated the order records. They show no real restaurant, kitchen, robot, customer or person. Like the code, they are released under the MIT License (see [LICENSE](../LICENSE)).

| Files | What they are | Used by |
|---|---|---|
| `videos/download.mp4`, `videos/download (1).mp4` … `videos/download (31).mp4` (32 files, 54 MB) | Food-preparation clips. MP4 (H.264), 1280×720, 24 fps, 5.0 s each | `scripts/01_embed_media.py` → the `media` and `frames` tables |
| `images/Crispy_Shoyu.jpg`, `House_Salmon.jpg`, `Kid's_Crab_Bowl.jpg`, `Miso_Chicken.jpg`, `Miso_Salmon.jpg`, `Salmon_Edamame.jpg`, `Tuna_Lover.jpg` (7 files) | Reference photos of finished bowls, 2500×1667 | `scripts/01_embed_media.py` → the `media` table (`media_type = 'image'`) |
| `orders/multirestaurant_orders.jsonl` (500 records) | Invented delivery orders for three fictional restaurants (Sample Seafood House, Sample Poke Bar, Sample Bowl Kitchen): items, prices, discounts, timings, ratings and reviews | Not loaded yet (see TODO.md); an example of the order data that sits next to the media in this pipeline |

## Generated (not in git)

`scripts/01_embed_media.py` writes everything derived from these files into `byteeats.lancedb/` at the repository root: the media bytes, thumbnails, metadata and embeddings, in the `media` and `frames` tables. Delete that folder (`make clean`) to start over. Nothing is written into `data/`.
