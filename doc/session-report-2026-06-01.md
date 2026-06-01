# Session Report — 2026-06-01 (Ship C)

**Goal:** Build Ship C — chunking layer + chunk-level ChromaDB indexing.
**Outcome:** ✅ All Ship C tasks complete. Unit tests pass (9/9). Smoke test
passes (run by user in venv: cleanup + `python -m tests.smoke_ship_c`).

## What got done

Ship C replaces whole-article embeddings with ~256-token **chunk-level** ones, so
retrieval (Ship D) operates on passage-sized vectors. ChromaDB now holds one entry
per chunk, keyed `{article_id}:{chunk_index}`, with `article_id` in metadata so the
eval harness (Ship F) can map a chunk back to its source article. The `indexed`
flag (added in Ship B) now gates re-runs so articles aren't re-chunked.

The daily-briefing path is untouched — it summarizes in-memory article dicts and
never read ChromaDB, so the conversion broke nothing downstream.

## Files created

| File | Purpose |
|---|---|
| `src/rag/__init__.py` | new `rag` package marker |
| `src/rag/chunker.py` | `chunk_article()` — token-windowed passage splitting |
| `tests/test_chunker.py` | 9 unit tests (deterministic FakeTokenizer); all pass |
| `tests/smoke_ship_c.py` | end-to-end smoke: structure + retrieval + indexed-gate |
| `doc/project-overview.md` | refresher: architecture, file map, runtime flow, schema |
| `doc/session-report-2026-06-01.md` | this report |

## Files modified

| File | Change |
|---|---|
| `src/config.py` | added `CHUNK_SIZE_TOKENS=256`, `CHUNK_OVERLAP_TOKENS=38` |
| `src/storage/database.py` | added `get_unindexed_articles()` + `mark_indexed()` (commits) |
| `src/storage/vector_store.py` | `add_articles` → `add_chunks(chunks, embeddings)`; updated `__main__` demo |
| `src/pipeline.py` | import `chunk_article`; rewired step 8 → chunk + embed + `add_chunks` + `mark_indexed` |

## Key decisions

- **Tokenizer passed in, not loaded inside** the chunker — reuses
  `embedder.model.tokenizer` so the model loads once and token counts match the
  embedder. `chunk_article(article, tokenizer, chunk_size, overlap)`.
- **Tokenizer param typed `Any`, not `PreTrainedTokenizerBase`** — the chunker is
  duck-typed (only needs `__call__` + `decode`). Avoids importing `transformers`
  just for an annotation (that import crashed on load where the package was
  absent) and lets the test's `FakeTokenizer` through. (Optional future: a
  `typing.Protocol` to document the contract without the heavy import.)
- **`add_special_tokens=False`** when counting — counts content tokens only; the
  model adds its own `[CLS]`/`[SEP]` at encode time.
- **`mark_indexed` on every fetched article**, even ones that yield zero chunks,
  so re-runs don't keep retrying them.
- **Ordering** — chunking runs *after* `save_articles` (chunk ids need the SQLite
  autoincrement `id`); `mark_indexed` commits so the gate persists.

## Operational steps performed

- Reset ChromaDB: `Remove-Item -Recurse -Force .\data\chroma` (one-time; wiped the
  stale URL-keyed article-level collection so it rebuilt chunk-level).
- First post–Ship C run re-chunked the existing back catalog (all rows had
  `indexed=False`).

## Status & next

- **Ship C: DONE.** Consider ticking the boxes in `doc/ship-c-chunking.md` and
  flipping Ship C → Done in `IMPLEMENTATION_PLAN.md` roadmap (not yet edited).
- **Next: Ship D** (Week 1, Jun 1–7) — retriever wrapper over `search_similar`
  (query embed + top-k), `GroundedAnswer` schema, cited-answer generation,
  Streamlit skeleton. Once it lands, add the screenshot to the README and the
  June-1 link becomes demo-backed.
