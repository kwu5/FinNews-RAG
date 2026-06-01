# Implementation Plan — FinNews-RAG (Project B)

**Direction:** Evolve the working MVP into **FinNews-RAG** — a retrieval-augmented
financial-news Q&A system whose headline differentiator is a **self-evaluation
harness** (retrieval precision/recall, faithfulness, latency/cost across configs).
**References:** `PROJECT_B_README.md` (vision), `financial_news_system_design_notes.md` (design notes)

## Goal

Turn the daily-summary MVP into a system that:
1. Ingests financial news from 8–10 free RSS feeds (World News API secondary).
2. **Chunks** articles and indexes chunk-level embeddings in ChromaDB.
3. **Retrieves** top-k chunks for a user query and generates a **grounded,
   source-cited answer** constrained to the retrieved context.
4. **Evaluates** itself — retrieval P/R on a hand-labeled set, RAGAS faithfulness
   and answer-relevance, latency/cost — and compares configurations.
5. Retains the daily briefing report as a secondary feature.

## Decisions (locked in)

| Decision | Choice | Rationale |
|---|---|---|
| Product direction | RAG Q&A + eval harness = headline; daily briefing = secondary | Defensible portfolio angle; pull model over push |
| Scope | Evolve in place (same repo) | Pipeline already works — reuse, don't rebuild |
| Orchestration framework | None — raw OpenAI / ChromaDB / sentence-transformers | Shows internals; avoids rewrite churn |
| Evaluation | Hybrid: RAGAS (faithfulness, answer-relevance) + custom (retrieval P/R, latency/cost) | Recognized metrics fast + ground-truth retrieval metrics |
| Demo UI | Streamlit | Fast to build, screenshots well |
| RSS strategy | RSS-first, 8–10 free/open feeds | Source diversity drives retrieval quality |
| World News API | Secondary fallback (backfill / low RSS yield) | Not primary |
| Paywalled feeds (FT, WSJ, Bloomberg) | Excluded | Extraction failure rate too high |
| Vector store | ChromaDB (persistent) | Already built |
| Embeddings | `all-MiniLM-L6-v2` | Local, free — also an eval comparison axis |
| Generation LLM | OpenAI `gpt-4o-mini` | Cost-effective |
| Chunking | ~256 tokens, ~15% overlap | Chunk size is an eval comparison axis |
| Test set | 50–100 query/relevant-doc pairs, **labels hand-verified** | Ground truth for retrieval metrics |
| Test-set granularity | Relevance labeled at **article** level | A retrieved chunk counts as relevant if its `article_id` is labeled relevant |
| DB migration | Wipe + recreate `data/news.db`; `output/*.md` preserved | Cleanest |
| Clustering (old v2 stage) | **Deferred / optional** | Not needed for query-time RAG |
| Hallucination check (old v2 Week 4) | **Folded into eval harness faithfulness metric** | Don't build it twice |
| Alembic | Not used | One-time wipe |
| Tiered hot/warm/cold retention | Deferred | Storage small (<200 MB / year) |

## What carries over vs. what's new

| Layer | Status |
|---|---|
| Config, RSS + World News ingestion, cleaner/NER, embeddings, SQLite + ChromaDB | Reuse |
| Dedup (Stages 1–2: URL canon + content hash) | Keep — prevents duplicate chunks polluting retrieval/eval |
| Daily briefing summarizer + report generator | Keep as secondary feature, unchanged |
| Chunking layer | New |
| Query-time retriever (top-k) | New |
| Grounded, source-cited Q&A | New |
| Evaluation harness + labeled test set + multi-config runner | New |
| Streamlit demo (`app.py`) | New |

New code areas: `src/rag/` (chunker, retriever, qa), `src/evaluation/` (harness,
metrics, testset), `app.py` (Streamlit), `evaluate.py` (eval entry point),
`eval/testset.jsonl` (labeled set).

## Schema (v2 + RAG)

```
articles  (SQLite)
  id, url, canonical_url (UNIQUE), content_hash (UNIQUE),
  title, description, content, source,
  published_at (UTC ISO 8601), fetched_at,
  extraction_method ('trafilatura' | 'newspaper3k' | 'readability' | 'rss-only'),
  indexed   (bool)   -- chunked + embedded into ChromaDB
  processed (bool)   -- included in a daily briefing

daily_reports  (SQLite)
  id, report_date (UNIQUE), content, article_count, created_at
  -- retained forever

ChromaDB collection "financial_news"
  one entry per CHUNK; id = "{article_id}:{chunk_index}"
  metadata: article_id, chunk_index, title, source, url, published_at
  document: chunk text
```

Dropped from the old v2 schema: `cluster_id` column and the `clusters` table
(clustering deferred). No SQLite `chunks` table — chunk→article mapping lives in
ChromaDB metadata; eval maps retrieved chunks to `article_id`.

## Structured output schemas

```python
# Grounded Q&A — lock before Ship D
class Citation(BaseModel):
    marker: int          # the [1], [2]... number used inline in `answer`
    chunk_id: str        # "{article_id}:{chunk_index}"
    article_id: int
    url: str

class GroundedAnswer(BaseModel):
    answer: str                   # inline [n] markers reference `citations`
    citations: list[Citation]
    answered_from_context: bool   # False -> retrieved context was insufficient
```

The daily-briefing structured model is unchanged from the MVP.

## Roadmap — ordered ships

Scheduling lives in `doc/june-weekly-schedule.md`, not here. This table is order
+ status only.

| Ship | Focus | Status |
|---|---|---|
| A | Ingestion overhaul — RSS-first, full-text extraction, source health | Done |
| B | Dedup Stages 1–2 (URL canon + content hash); wipe + recreate DB | Done |
| C | Chunking layer + chunk-level embeddings indexed in ChromaDB | **Current** |
| D | Retriever + grounded cited Q&A; Streamlit skeleton | Planned |
| E | Labeled test set (50–100 query/relevant-doc pairs) | Planned |
| F | Eval harness pt.1 — retrieval precision/recall + latency/cost | Planned |
| G | Eval harness pt.2 — RAGAS faithfulness + answer-relevance | Planned |
| H | Multi-config comparison runner + written findings | Planned |
| I | Streamlit polish + README finalize; stretch: signal extraction | Planned |

**Cadence rule:** at each ship boundary, re-read this file and rewrite only the
next ship's detail section to full resolution. Keep one ship detailed at a time.
Scheduling is tracked separately in `doc/june-weekly-schedule.md`.

### Ship C — Chunking layer + chunk-level ChromaDB (CURRENT)

**Full detail:** `doc/ship-c-chunking.md` (this section is the master-plan
summary; the doc is the working copy with watch-outs).

**Goal:** Replace article-level embeddings with chunk-level ones so retrieval
(Ship D) operates on ~paragraph-sized passages instead of whole articles. Index
each chunk in ChromaDB keyed `{article_id}:{chunk_index}`, carrying `article_id`
in metadata so the eval harness (Ship F) can map a retrieved chunk back to its
source article.

**Why now:** Retrieval quality is the foundation everything downstream sits on.
Whole-article embeddings dilute the signal — a query about one earnings figure
shouldn't have to match against a 1,500-word article vector. This is also where
the `indexed` flag (added in Ship B) finally gets used.

**Resolved up front:**
- The daily briefing does NOT depend on ChromaDB — it summarizes the in-memory
  `articles` dicts directly. Today's article-level embeddings are write-only
  (nothing reads them back; dedup Stage 3 embeds internally). So the chunk-level
  conversion breaks nothing in the briefing path — no dual embedding scheme.
- `all-MiniLM-L6-v2` truncates at 256 tokens, so ~256-token chunks match the
  model's actual capacity rather than being an arbitrary choice.

#### Tasks

- [ ] **Config** (`src/config.py`) — add `CHUNK_SIZE_TOKENS: int = 256` and
      `CHUNK_OVERLAP_TOKENS: int = 38` (~15%); both become eval axes in Ship H.
- [ ] **New module `src/rag/chunker.py`** —
      `chunk_article(article: dict, chunk_size: int, overlap: int) -> list[dict]`.
      Chunk dict: `{chunk_index, text, article_id, title, source, url,
      published_at}`. Count tokens with the embedding model's tokenizer; short
      articles → one chunk; sequential `chunk_index` from 0; overlap carries
      trailing tokens forward.
- [ ] **Convert `vector_store.py`** — replace `add_articles` with
      `add_chunks(chunks, embeddings)`: id `f"{article_id}:{chunk_index}"`,
      document = chunk text, metadata = `article_id / chunk_index / title /
      source / url / published_at`. `search_similar` unchanged (now returns
      chunk hits).
- [ ] **Database accessors** (`src/storage/database.py`) — add
      `get_unindexed_articles()` (`indexed == False`) and
      `mark_indexed(article_ids)`. Chunk IDs need the autoincrement `id`, only
      available after `save_articles`.
- [ ] **Rewire pipeline** (`src/pipeline.py`) — replace the article-level embed
      block with: `save_articles` → `get_unindexed_articles()` → chunk → embed
      chunks → `add_chunks` → `mark_indexed`. Briefing step untouched.
- [ ] **Reset ChromaDB** — delete `data/chroma/` (stale URL-keyed article-level
      entries) so it rebuilds chunk-level. SQLite is NOT wiped; existing rows
      have `indexed=False` so the first run re-chunks the back catalog.
- [ ] **Tests** — chunker unit tests (short → 1 chunk; long → N with correct
      overlap + sequential indices; metadata carried).
- [ ] **Smoke test** (`tests/smoke_ship_c.py`) — ChromaDB count > article count;
      IDs match `{int}:{int}`; every chunk carries `article_id`; second same-day
      run does not re-index.

**Done when:** ChromaDB holds chunk-level entries with `article_id` in metadata;
the `indexed` flag flips and gates re-runs; `search_similar` returns passage
hits; the daily briefing still generates unchanged.

**Watch-outs:** `get_unindexed_articles()` must run *after* `save_articles`
(chunk IDs need the DB `id`); re-run safety relies on `mark_indexed` committing,
not on ChromaDB upsert idempotency.

**Deferred to Ship D:** retriever wrapper over `search_similar` (query embed +
top-k), the `GroundedAnswer` schema, cited-answer generation, Streamlit skeleton.

## Risk register

| Risk | Mitigation |
|---|---|
| Trafilatura/newspaper3k flaky on some sites | Three-tier fallback; tolerate ~10% extraction failures, exclude those articles |
| Chunk size too small loses context / too large dilutes retrieval | Chunk size is an eval comparison axis; start at 256 tokens |
| Test set too small to be meaningful | 50–100 pairs minimum; hand-verify every relevance label |
| RAGAS LLM-as-judge calls add cost | Run on a subset; cache judge outputs; budget per eval run |
| OpenAI structured-output rejects schema | Keep `GroundedAnswer` flat; test with small fixtures first |
| Config tuning eats more than allotted time | Hard-cap Ship H; ship with defaults (chunk 256, top-k 5) if inconclusive |
| Scope creep into clustering / tiered retention | Explicitly deferred — re-read this doc if tempted |

## Out of scope for FinNews-RAG v1

- Cross-day topic clustering and weekly briefings.
- Tiered hot/warm/cold retention (defer until DB > 1 GB).
- Paywalled feeds (FT, WSJ, Bloomberg).
- Finnhub/Marketaux replacement for World News API.
- Real-time market data integration.
- Email/Slack delivery, multi-language.
- Backtest of sentiment vs. price movement (README stretch goal only).
