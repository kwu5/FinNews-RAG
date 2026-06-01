# Project Overview — FinNews-RAG

**Purpose of this doc:** a single refresher on what the program is, how it's
structured, and how data flows through it at runtime. Reflects the code as it
actually exists today (mid–Ship C). For *direction* see `IMPLEMENTATION_PLAN.md`;
for *schedule* see `doc/june-weekly-schedule.md`.

---

## 1. What this project is

It started as a **daily financial-news summarizer** (ingest news → dedupe →
LLM briefing → Markdown/HTML report) and is mid-pivot into **FinNews-RAG**: a
retrieval-augmented Q&A system over the same news corpus, whose headline feature
is a **self-evaluation harness** (retrieval precision/recall, faithfulness,
latency/cost across configs).

Two product surfaces, sharing one ingestion/storage backbone:

| Surface | Status | What it does |
|---|---|---|
| **Daily briefing** | ✅ Working (the original MVP) | Fetch → clean → dedupe → summarize → export a dated report |
| **RAG Q&A + eval** | 🚧 Being built (Ships C–I) | Chunk → embed → retrieve top-k → cited answer → measure quality |

The briefing path summarizes in-memory article dicts directly via the LLM — it
does **not** read from ChromaDB. That's why the Ship C conversion of ChromaDB
from article-level to chunk-level breaks nothing in the briefing.

---

## 2. Layered architecture

```
                 ┌─────────────────────────────────────────────┐
   entry points  │  main.py  (--mode api | run-once | scheduler)│
                 └───────────────┬──────────────────┬──────────┘
                                 │                  │
                   ┌─────────────▼──────┐   ┌───────▼────────────┐
   faces           │ src/api/main.py    │   │ src/scheduler/     │
                   │ FastAPI endpoints  │   │ jobs.py (cron)     │
                   └─────────────┬──────┘   └───────┬────────────┘
                                 │                  │
                                 └────────┬─────────┘
                                          ▼
                              ┌───────────────────────┐
   orchestration             │  src/pipeline.py       │
                             │  run_pipeline()        │
                             └───────────┬───────────┘
            ┌──────────────┬─────────────┼────────────┬──────────────┐
            ▼              ▼             ▼            ▼              ▼
        ingestion     processing     storage     summarization    rag/ (new)
        RSS + WNA    clean/dedupe   SQLite +      LLM + report    chunker → …
                     /embeddings    ChromaDB                      retriever,qa
```

Key idea: **`pipeline.py` is the single orchestrator.** Both the API and the
scheduler call `run_pipeline()`; nothing duplicates the fetch→export sequence.
Heavy objects (embedding model, DB, vector store, LLM client) are instantiated
**once at module import** in `pipeline.py`, so models load a single time.

---

## 3. Program structure (what each file does)

```
main.py                         Entry point. argparse → api | run-once | scheduler
src/
├── config.py                   Pydantic Settings — env-driven config (keys, paths,
│                                 thresholds, CHUNK_SIZE/OVERLAP coming in Ship C)
├── pipeline.py                 run_pipeline(): the full ingest→summarize→export flow
│
├── ingestion/
│   ├── rss_reader.py           PRIMARY source. Parses feeds from config/feeds.yaml,
│   │                             normalizes articles, tracks per-feed health
│   ├── extractor.py            Full-text fallback chain: trafilatura → newspaper3k
│   │                             → readability; degrades to rss-only
│   ├── world_news_api.py       FALLBACK source. Used only when RSS yield is low
│   └── search_news_sample.py   (sample/scratch)
│
├── processing/
│   ├── cleaner.py              TextCleaner — strip URLs/emails/ads, normalize, NER
│   ├── url_canon.py            canonicalize_url() — dedup stage 1 key
│   ├── content_hash.py         compute_content_hash() — dedup stage 2 key
│   ├── deduplicator.py         Deduplicator — embedding-similarity dedup (stage 3)
│   └── embeddings.py           EmbeddingGenerator — wraps all-MiniLM-L6-v2
│
├── storage/
│   ├── database.py             SQLAlchemy. Article + DailyReport models, Database CRUD
│   └── vector_store.py         ChromaDB wrapper (add_articles today → add_chunks in C)
│
├── summarization/
│   ├── llm_client.py           LLMClient — GPT-4o-mini briefing + sentiment
│   └── report_generator.py     ReportGenerator — Markdown + Jinja2 HTML export
│
├── rag/                        ← NEW (Ship C+)
│   └── chunker.py              chunk_article() — token-windowed passage splitting
│
├── api/main.py                 FastAPI: / · /health · /generate · /report/{date} · /reports
└── scheduler/jobs.py           APScheduler — daily run at DAILY_RUN_HOUR
```

---

## 4. Runtime workflow — the daily pipeline

This is `run_pipeline()` in `src/pipeline.py`, step by step (current code):

1. **Fetch** — `rss.fetch_from_feeds()` first. If yield < `RSS_YIELD_THRESHOLD`
   (15), top up from `news_api.fetch_financial_news()`. Empty → raise.
2. **Validate** — drop articles missing `title` or `content`.
3. **Canonicalize URL** — `canonicalize_url(url)` per article; drop unparseable
   (dedup key #1).
4. **Clean** — `cleaner.clean_article(content)` in place (strip ads/URLs/whitespace).
5. **Content hash** — `compute_content_hash()` *after* cleaning (dedup key #2).
6. **Deduplicate** — `dedup.deduplicate_articles()` (embedding similarity).
7. **Persist** — `db.save_articles()` (skips rows whose url/canonical_url/
   content_hash already exist).
8. **Embed + index** — *today:* article-level embeddings → `vstore.add_articles()`.
   *Ship C target:* `get_unindexed_articles()` → chunk each → embed chunks →
   `add_chunks()` → `mark_indexed()`.
9. **Summarize** — `llm.generate_summary(articles)` → briefing markdown.
10. **Export** — `reporter.save_markdown()` + `generate_html()` →
    `output/financial_briefing_YYYY-MM-DD.{md,html}`.
11. **Record** — `db.save_report()` upserts the DailyReport row for today.

Returns `(summary_markdown, article_count)`.

### How it's triggered (`main.py --mode`)
- **`api`** — starts the scheduler thread *and* a uvicorn server. `POST /generate`
  runs the pipeline on demand (short-circuits to today's cached report unless
  `force_refresh`).
- **`run-once`** — one synchronous `run_pipeline()` and exit. Good for testing.
- **`scheduler`** — scheduler only; blocking loop, fires daily at `DAILY_RUN_HOUR`.

---

## 5. Data model

### SQLite (`data/news.db`) — source of record for articles
```
articles
  id              INTEGER PK autoincrement   ← becomes the chunk article_id
  title, description, content
  content_hash    UNIQUE                     dedup key #2
  url             UNIQUE
  canonical_url   UNIQUE                     dedup key #1
  source, published_at (UTC ISO), fetched_at
  extraction_method   trafilatura|newspaper3k|readability|rss-only
  processed       bool   — included in a daily briefing
  indexed         bool   — chunked+embedded into ChromaDB (gates Ship C re-runs)

daily_reports
  id, report_date UNIQUE, content, article_count, created_at
```

### ChromaDB (`data/chroma/`) — vector index
- **Today:** one entry per *article*, keyed by URL (write-only — nothing reads
  it back).
- **Ship C target:** one entry per *chunk*, id `"{article_id}:{chunk_index}"`,
  metadata `article_id / chunk_index / title / source / url / published_at`,
  document = chunk text. This is what the retriever (Ship D) and eval harness
  (Ship F) will query.

---

## 6. Where Ship C fits

Ship C swaps the embedding granularity from whole-article to ~256-token chunks
so retrieval works on passage-sized vectors. Touch points:

- `config.py` — add `CHUNK_SIZE_TOKENS=256`, `CHUNK_OVERLAP_TOKENS=38`
- `rag/chunker.py` — `chunk_article()` ✅ drafted
- `database.py` — add `get_unindexed_articles()` + `mark_indexed()`
- `vector_store.py` — `add_articles` → `add_chunks`
- `pipeline.py` — rewire step 8 (⚠ order: `save_articles` *before*
  `get_unindexed_articles` — chunk IDs need the autoincrement `id`)
- one-time: delete `data/chroma/` so it rebuilds chunk-level

**Done when:** ChromaDB holds chunk entries with `article_id` in metadata, the
`indexed` flag flips + gates re-runs, and the daily briefing still generates
unchanged.

---

## 7. Roadmap at a glance

| Ship | Focus | Status |
|---|---|---|
| A | RSS-first ingestion + full-text extraction + source health | ✅ |
| B | Dedup stages 1–2 (URL canon + content hash) | ✅ |
| **C** | **Chunking + chunk-level ChromaDB** | **🚧 current** |
| D | Retriever + grounded cited Q&A + Streamlit | planned |
| E | Labeled test set (50–100 pairs) | planned |
| F | Eval harness pt.1 — retrieval P/R + latency/cost | planned |
| G | Eval harness pt.2 — RAGAS faithfulness + answer-relevance | planned |
| H | Multi-config comparison runner + written findings | planned |
| I | Streamlit polish + README finalize | planned |

Goal: Ships C–I done by end of June, one ship per week (see
`doc/june-weekly-schedule.md`).
```
