# Ship C — Chunking layer + chunk-level ChromaDB

**Status:** Done (2026-06-01)
**Drafted:** 2026-05-29
**Parent plan:** `IMPLEMENTATION_PLAN.md` (roadmap ship C)
**Predecessors:** Ship A (ingestion overhaul) ✅, Ship B (dedup Stages 1–2) ✅

## Goal

Replace article-level embeddings with chunk-level ones so retrieval (Ship D)
operates on ~paragraph-sized passages instead of whole articles. Index each
chunk in ChromaDB keyed `{article_id}:{chunk_index}`, carrying `article_id` in
metadata so the eval harness (Ship F) can map a retrieved chunk back to its
source article.

## Why now

Retrieval quality is the foundation everything downstream sits on. Whole-article
embeddings dilute the signal — a query about one earnings figure shouldn't have
to match against a 1,500-word article vector. This is also where the `indexed`
flag (added in Ship B) finally gets used.

## Resolved up front

- **The daily briefing does NOT depend on ChromaDB.** In `pipeline.py` the
  briefing summarizes the in-memory `articles` dicts directly
  (`llm.generate_summary(articles)`). The article-level embeddings currently
  pushed to ChromaDB are effectively write-only — nothing reads them back
  (dedup Stage 3 generates its own embeddings internally). So converting
  ChromaDB to chunk-level breaks nothing in the briefing path. No dual
  embedding scheme to maintain.
- **`all-MiniLM-L6-v2` truncates at 256 tokens.** A ~256-token chunk size isn't
  arbitrary — it's the model's actual capacity. Chunks larger than that would be
  silently truncated, so chunking is what lets us embed full articles without
  losing the tail.

## Tasks

- [x] **Config** (`src/config.py`) — add `CHUNK_SIZE_TOKENS: int = 256` and
      `CHUNK_OVERLAP_TOKENS: int = 38` (~15%). These become eval comparison axes
      in Ship H, so make them settings, not constants.
- [x] **New module `src/rag/chunker.py`** —
      `chunk_article(article: dict, chunk_size: int, overlap: int) -> list[dict]`.
      Each returned chunk dict: `{chunk_index, text, article_id, title, source,
      url, published_at}`. Count tokens with the embedding model's own tokenizer
      (not word-splitting) for accuracy against the 256 cap. Short articles →
      one chunk; sequential `chunk_index` from 0; overlap carries trailing
      tokens into the next chunk.
- [x] **Convert `vector_store.py`** — replace `add_articles` with
      `add_chunks(chunks: list[dict], embeddings: list)`: id =
      `f"{c['article_id']}:{c['chunk_index']}"`, document = chunk text,
      metadata = `article_id / chunk_index / title / source / url /
      published_at` (note: `url` and `article_id` are new to the metadata vs.
      today). Keep `search_similar` as-is — it now returns chunk hits.
- [x] **Database accessors** (`src/storage/database.py`) — add
      `get_unindexed_articles() -> list` (`indexed == False`) and
      `mark_indexed(article_ids: list)`. Needed because chunk IDs require the
      SQLite autoincrement `id`, which only exists after `save_articles`.
- [x] **Rewire pipeline** (`src/pipeline.py`) — replace the article-level embed
      block (lines 97–100) with: `save_articles` → `get_unindexed_articles()` →
      chunk each → embed all chunks → `add_chunks` → `mark_indexed`. Only
      unindexed articles get chunked, so re-runs don't duplicate. Briefing step
      stays untouched on the in-memory dicts.
- [x] **Reset ChromaDB** — the existing collection holds stale article-level
      entries keyed by URL. One-time: delete `data/chroma/` (or reset the
      collection) so it rebuilds chunk-level. SQLite is **not** wiped (the
      `indexed` column already exists from Ship B) — but every existing row has
      `indexed=False`, so the first run re-chunks the whole back catalog, which
      is what we want.
- [x] **Tests** (`tests/test_processing.py` or new `tests/test_chunker.py`) —
      short article → exactly 1 chunk; long article → N chunks with correct
      overlap and sequential indices; metadata carried through.
- [x] **Smoke test** (`tests/smoke_ship_c.py`) — run pipeline; assert ChromaDB
      count > article count, all IDs match `{int}:{int}`, every chunk has
      `article_id` in metadata; run a sample query and eyeball that hits are
      passage-sized. Second run on the same day → no re-indexing (indexed flag
      holds).

## Done when

ChromaDB holds chunk-level entries with `article_id` in metadata; the `indexed`
flag flips after indexing and gates re-runs; `search_similar` returns passage
hits; the daily briefing still generates unchanged.

## Deferred to Ship D

A thin retriever wrapper over `search_similar` (query embed + top-k), the
`GroundedAnswer` schema, cited-answer generation, and the Streamlit skeleton.

## Watch-outs

- **Ordering:** `get_unindexed_articles()` must run *after* `save_articles` —
  chunk IDs need the SQLite autoincrement `id`. Reversing this is the one wiring
  subtlety that will bite.
- Re-runs rely on the `indexed` flag, not on ChromaDB upsert idempotency. Make
  sure `mark_indexed` actually commits, or every run re-chunks everything.
