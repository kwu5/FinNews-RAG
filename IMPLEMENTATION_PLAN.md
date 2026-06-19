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
| C | Chunking layer + chunk-level embeddings indexed in ChromaDB | Done |
| D | Retriever + grounded cited Q&A; Streamlit skeleton | **Done** |
| E | Labeled test set (50–100 query/relevant-doc pairs) | **Done** |
| F | Eval harness pt.1 — retrieval precision/recall + latency/cost | **Done** |
| G | Eval harness pt.2 — faithfulness + answer-relevance (custom LLM-judge) | **In progress** |
| H | Multi-config comparison runner + written findings | Planned |
| I | Streamlit polish + README finalize; **audit LLM-labeled testset half (Ship E)**; stretch: signal extraction | Planned |

**Cadence rule:** at each ship boundary, re-read this file and rewrite only the
next ship's detail section to full resolution. Keep one ship detailed at a time.
Scheduling is tracked separately in `doc/june-weekly-schedule.md`.

### Ship C — Chunking layer + chunk-level ChromaDB (DONE)

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

- [x] **Config** (`src/config.py`) — add `CHUNK_SIZE_TOKENS: int = 256` and
      `CHUNK_OVERLAP_TOKENS: int = 38` (~15%); both become eval axes in Ship H.
- [x] **New module `src/rag/chunker.py`** —
      `chunk_article(article: dict, chunk_size: int, overlap: int) -> list[dict]`.
      Chunk dict: `{chunk_index, text, article_id, title, source, url,
      published_at}`. Count tokens with the embedding model's tokenizer; short
      articles → one chunk; sequential `chunk_index` from 0; overlap carries
      trailing tokens forward.
- [x] **Convert `vector_store.py`** — replace `add_articles` with
      `add_chunks(chunks, embeddings)`: id `f"{article_id}:{chunk_index}"`,
      document = chunk text, metadata = `article_id / chunk_index / title /
      source / url / published_at`. `search_similar` unchanged (now returns
      chunk hits).
- [x] **Database accessors** (`src/storage/database.py`) — add
      `get_unindexed_articles()` (`indexed == False`) and
      `mark_indexed(article_ids)`. Chunk IDs need the autoincrement `id`, only
      available after `save_articles`.
- [x] **Rewire pipeline** (`src/pipeline.py`) — replace the article-level embed
      block with: `save_articles` → `get_unindexed_articles()` → chunk → embed
      chunks → `add_chunks` → `mark_indexed`. Briefing step untouched.
- [x] **Reset ChromaDB** — delete `data/chroma/` (stale URL-keyed article-level
      entries) so it rebuilds chunk-level. SQLite is NOT wiped; existing rows
      have `indexed=False` so the first run re-chunks the back catalog.
- [x] **Tests** — chunker unit tests (short → 1 chunk; long → N with correct
      overlap + sequential indices; metadata carried).
- [x] **Smoke test** (`tests/smoke_ship_c.py`) — ChromaDB count > article count;
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

### Ship D — Retriever + grounded cited Q&A + Streamlit skeleton (DONE)

**Full detail:** `doc/ship-d-retriever-qa.md` (this section is the master-plan
summary; the doc is the working copy with watch-outs).

**Goal:** Turn Ship C's chunk index into a query-time RAG path — retrieve top-k
chunks for a question and generate a **grounded answer constrained to that
context** with inline `[n]` citations that resolve to real source articles,
surfaced through a minimal Streamlit app. This is **the demo**; its screenshot is
what makes the posted link demo-backed (`doc/june-weekly-schedule.md`, Week 1).

**Why now:** Ship C made retrieval possible (passage-sized vectors with
`article_id` in metadata); Ship D makes it usable. Ships E–H all evaluate the
`answer_query()` seam, so it must exist first.

**Resolved up front:**
- **Citations are built from the retrieved chunks, not the LLM.** The model cites
  by source *number*; we map numbers back to the authoritative
  `chunk_id`/`article_id`/`url` from retrieval. The model can mis-number a
  citation but cannot fabricate its target.
- **No new DB accessors** — ChromaDB chunk metadata already carries everything
  citations need (`article_id`, `url`, `title`, `published_at`).
- **ChromaDB query results are nested one level** (`results["ids"][0]`…); the
  retriever flattens this so `[0]` never leaks downstream.
- The daily-briefing path is untouched; Ship D adds a parallel `src/rag/` path.

#### Tasks

- [x] **Config** (`src/config.py`) — add `RETRIEVAL_TOP_K: int = 5` (eval axis in
      Ship H).
- [x] **`src/rag/retriever.py`** — `retrieve(query, top_k) -> list[dict]`: embed
      query → `search_similar` → flatten to hit dicts `{chunk_id, article_id,
      text, title, source, url, published_at, distance}`; empty → `[]`.
- [x] **`src/rag/qa.py`** — `Citation` / `GroundedAnswer` schemas +
      `answer_query(query, top_k) -> GroundedAnswer`: retrieve → empty-hit
      short-circuit (no LLM call, `answered_from_context=False`) → numbered
      context block → grounded generation → build citations from `hits` by marker.
      `Citation` carries `title`/`source`/`url` (decision (b)) so it's
      self-contained for the UI + eval.
- [x] **`LLMClient.generate_grounded_answer`** — OpenAI structured output
      (`parse`, flat response model: `answer`, `used_markers`,
      `answered_from_context`), `temperature=0`, "answer only from context" prompt.
- [x] **`app.py`** — Streamlit: query box → `answer_query` → answer + Sources
      list; minimal styling (polish is Ship I).
- [x] **Tests** — retriever flatten/empty; qa empty-hits no-LLM-call, citation
      marker resolution, out-of-range marker dropped. (9 unit tests, green.)
- [x] **Smoke test** (`tests/smoke_ship_d.py`) — in-domain query → cited answer
      whose `url` matches a hit; out-of-domain → `answered_from_context=False`.

**Done when:** a finance question in Streamlit returns a grounded, inline-cited
answer with a resolved source list; out-of-domain/nothing-retrieved reports
insufficient context instead of fabricating; `answer_query()` is importable as the
seam Ship F will evaluate.

**Watch-outs:** flatten the ChromaDB `[0]` once in the retriever; build citations
from retrieval not the LLM; handle the zero-hit case before any LLM call;
`temperature=0` for eval reproducibility.

**Deferred to later ships:** labeled test set (E); retrieval P/R + latency/cost
(F); RAGAS faithfulness/answer-relevance (G); multi-config sweep (H); distance
relevance-floor + Streamlit polish (I).

### Ship E — Labeled test set (50–100 query/relevant-article pairs) (DONE)

**Full detail:** `doc/ship-e-testset.md` (this section is the master-plan summary;
the doc is the working copy with watch-outs).

**Goal:** Produce `eval/testset.jsonl` — 50–100 finance questions, each labeled
with the set of corpus **article ids** relevant to it. This is the hand-verified
ground truth Ships F–G score `answer_query()`/`retrieve()` against. Relevance is
labeled at the **article** level (a chunk is relevant iff its `article_id` is in
the query's `relevant_article_ids`).

**Why now:** Ship D made `answer_query()` real but unmeasured. The eval harness is
the headline differentiator, and every metric in F–H scores against this set —
without trustworthy labels there is nothing to measure. Hand-verification is the
deliverable's value, not overhead.

**Decisions (locked 2026-06-09):**
- **Hybrid query sourcing** — mostly LLM-generated from indexed articles (fast,
  guaranteed answerable), plus ~10 hand-written hard / out-of-domain queries.
- **Pooling + spot-check** — pool candidates via `retrieve()` at a high pool-depth
  (≈20–30, deliberately larger than `RETRIEVAL_TOP_K` to avoid biasing labels
  toward the system being evaluated), label them, then scan a few extras per query.
- **Small CLI helper** labels interactively (`y`/`n`/`s`) and writes the JSONL —
  resumable, so the multi-day manual job can stop and continue.

**Schema** (`eval/testset.jsonl`, one object per line): `query_id`, `query`,
`relevant_article_ids: list[int]` (`[]` for out-of-domain), `source: "llm"|"hand"`,
`type: "in_domain"|"out_of_domain"`, `notes`. Out-of-domain rows (a flagged ~5–10
minority) test Ship D's abstention path and stay out of the in-domain P/R averages.

#### Tasks (built by the user)

- [ ] `eval/gen_queries.py` — LLM-generate candidate questions from sampled DB
      articles → `eval/queries_candidates.jsonl` for human curation; dedup; then
      hand-add ~10 hard/out-of-domain queries.
- [ ] `eval/label_testset.py` — pool via `retrieve(query, POOL_DEPTH)`, dedup hits
      to unique `article_id`, prompt y/n/skip per candidate, spot-check pass, append
      to `eval/testset.jsonl`. **Resumable** (skip already-labeled `query_id`s).
      `POOL_DEPTH` is a local constant, not a `Settings` field.
- [ ] Tiny `get_articles_by_ids()` / sample accessor in `database.py` — only if the
      spot-check path needs articles outside the pool (pooled hits already carry
      title/source/text).
- [ ] `eval/validate_testset.py` (or pytest) — all lines parse; unique `query_id`s;
      every relevant id exists; out-of-domain ⇔ empty relevant set; 50–100 rows;
      in-domain rows have ≥1 relevant id.
- [ ] Commit `eval/testset.jsonl` (ground truth belongs in git; confirm
      `.gitignore` doesn't sweep `eval/`).

**Done when:** `eval/testset.jsonl` holds 50–100 hand-verified rows passing the
validation script, every relevant id resolves to a real article, the out-of-domain
subset is present and flagged, and the file is committed — Ship F can load it and
score retrieval with no further labeling.

**Watch-outs:** pooling bias (pool deep + spot-check; note union-pooling across
configs for Ship H); dedup chunks→articles when pooling; keep the helper resumable
and append-only; the model proposes queries but **you** decide relevance; keep
out-of-domain a flagged minority so it doesn't distort P/R.

**Labeling split (2026-06-17, time-boxed):** behind schedule, so done-first over
perfect — the in-domain queries are split in half: the assistant labels its half
directly from drafted suggestions, the user hand-verifies the other half (working
from the same drafts). Out-of-domain rows are mechanical (`[]`). The
assistant-labeled half — plus q010-style pooling misses where `retrieve()` never
surfaced the seed — is **flagged for audit in Ship I**, restoring full
hand-verification once there's time.

**Deferred:** retrieval P/R + latency/cost (F); RAGAS faithfulness/answer-relevance
(G); multi-config sweep + union-pooling (H); audit of the assistant-labeled testset
half (I).

### Ship F — Eval harness pt.1: retrieval P/R + latency/cost (DONE)

**Full detail:** `doc/ship-f-retrieval-eval.md` (this section is the master-plan
summary; the doc is the working copy with watch-outs).

**Goal:** Load `eval/testset.jsonl` and score `retrieve()` against it — report
retrieval **precision/recall (and MRR) at the article level**, the out-of-domain
**abstention** rate as a separate number, and **latency** per query. This is the
first half of the headline differentiator: the test set built in Ship E finally
gets used to put numbers on retrieval quality.

**Why now:** Ship E produced trustworthy ground truth; nothing scores against it
yet. Ship F turns `answer_query()`/`retrieve()` from "works in the demo" into
"measurably this good," and establishes the metric plumbing Ships G–H extend.

**Resolved up front:**
- **Article-level scoring.** A retrieved chunk counts as a hit iff its
  `article_id` ∈ the query's `relevant_article_ids`. Dedup chunks→articles before
  scoring so one article can't be counted twice (mirrors the Ship E pooling rule).
- **In-domain vs out-of-domain split.** P/R/MRR are computed over in-domain
  queries only. Out-of-domain rows (`relevant_article_ids == []`) are scored
  separately as abstention: did `answer_query()` return `answered_from_context=False`?
  Never fold OOD into the P/R averages (Ship E watch-out).
- **Cost ≈ free for retrieval.** Embedding is the local MiniLM model and ChromaDB
  is local, so the retrieval path has ~no API cost — report **latency**
  (embed + query, with percentiles) and note cost is N/A until the LLM-judge in
  Ship G. Use `temperature=0` paths already in place for reproducibility.
- **Report at `RETRIEVAL_TOP_K` but sweep a few k.** Headline numbers at the
  served `top_k=5`, plus P/R at k ∈ {1,3,5,10} so Ship H has a baseline curve.

**Tasks (built by the user):**
- [ ] `src/evaluation/testset.py` — loader: parse `eval/testset.jsonl` into typed
      rows (`query_id, query, relevant_article_ids, source, type, notes`);
      basic integrity asserts (reuse/much like `validate_testset.py`).
- [ ] `src/evaluation/metrics.py` — pure functions over (retrieved article_ids,
      relevant set): `precision_at_k`, `recall_at_k`, `mrr`; plus an
      `abstention_correct(answered_from_context, is_out_of_domain)` helper. No I/O.
- [ ] `src/evaluation/harness.py` — iterate the test set, call `retrieve()` (and
      `answer_query()` for OOD abstention), time each call, dedup chunk hits to
      articles, aggregate per-k metrics split by `type`, return a results object.
- [ ] `evaluate.py` — CLI entry point: `--top-k`, `--k-sweep`, `--out`; run the
      harness and write a Markdown/JSON report to `output/eval/`.
- [ ] Report: per-k precision/recall/MRR table, abstention accuracy, latency
      p50/p95, and an explicit note that Ship E's pooling misses bias recall
      (flagged for the Ship I audit).

**Done when:** `python evaluate.py` loads the committed test set and prints
article-level P/R/MRR at the served top-k (plus a small k-sweep), reports OOD
abstention accuracy separately, and reports per-query latency — with the
pooling-bias caveat stated in the output.

**Watch-outs:** dedup chunks→articles before scoring; keep OOD out of P/R; the
recall ceiling is bounded by Ship E's pooling (don't read low recall as purely a
retriever failure — some relevant articles were never pooled, hence the audit);
hold `top_k`/chunk params at defaults here — sweeping configs is Ship H, not F.

**Deferred:** RAGAS faithfulness/answer-relevance (G); multi-config sweep +
union-pooling (H); distance relevance-floor tuning (I).

### Ship G — Eval harness pt.2: faithfulness + answer-relevance (IN PROGRESS)

**Full detail:** `doc/ship-g-judge-eval.md` (this section is the master-plan
summary; the doc is the working copy with watch-outs).

**Goal:** Score the **generated answer** (not just retrieval) with an LLM-judge:
**faithfulness** (every claim in `answer` is supported by the retrieved context)
and **answer-relevance** (the answer actually addresses the query). Report mean
faithfulness / answer-relevance over the answered in-domain queries, plus a
token/cost summary. Output a Markdown + JSON report under `output/eval/`.

**Why now:** Ship F measured retrieval but never judged the answer text. This is
the second half of the headline differentiator and the home of the old v2
"hallucination check," which the plan folded into the eval harness (faithfulness
*is* that check, externalized and per-claim). It must follow F (reuses the
harness/CLI plumbing) and precede H (H sweeps configs and needs both metric halves
in place to compare).

**Decisions (locked 2026-06-18):**

| Decision | Choice | Rationale |
|---|---|---|
| Metric implementation | **Custom LLM-judge**, not the RAGAS library | RAGAS pulls in LangChain → conflicts with the locked "no orchestration framework" decision; we keep RAGAS's metric *definitions*, compute them ourselves with `LLMClient` + MiniLM |
| Faithfulness method | Claim decomposition → per-claim support check vs. context | RAGAS definition: `supported_claims / total_claims` |
| Answer-relevance method | Generate N=3 reverse-questions from the answer → mean cosine to the original query (MiniLM); noncommittal answer → 0 | RAGAS definition; reuses the local embedder, no extra API cost for the cosine step |
| Scope | **In-domain queries that were answered** (`answered_from_context=True`) only | OOD rows abstain (no answer to judge); already scored as abstention in Ship F |
| Cost control | Disk **cache** keyed by `(query_id, metric, hash(answer+context))`; `--gen-sample N` cap; print token + $ estimate | Risk register: subset, cache judge outputs, budget per run. gpt-4o-mini makes the full set cents, but cache buys free + deterministic re-runs |
| Determinism | Judge calls at `temperature=0`; cache locks replay | Judge is itself an LLM — note its variance as a caveat |
| Mutation | Read-only over DB + ChromaDB (writes only the judge cache + report) | Eval must not alter the corpus |

**Seams it consumes (do not modify):** `qa.answer_query()` (generates the answer
under test — costs an LLM call now), `Retriever.retrieve()` (rebuilds the context
block for the faithfulness check, free/deterministic), `LLMClient` (flat
structured-output pattern), `EmbeddingGenerator` (`normalize_embeddings=True`),
`eval/testset.jsonl` (in-domain rows), the Ship F harness/CLI.

**Metric definitions (as implemented):**
- **faithfulness** = `supported_claims / total_claims`. Decompose `answer` into
  atomic statements (1 LLM call); verify all statements against the numbered
  context in **one batched** call (verdict + reason per statement). No factual
  claims (e.g. pure abstention text) → **excluded** from the mean (report the
  count), not scored 1.0.
- **answer-relevance** = `mean(cosine(q_i, original_query))` over N=3 questions an
  LLM generates *from the answer* (1 LLM call); embed with MiniLM. If the
  generator flags the answer **noncommittal** ("I don't know"-style) → score 0.
- **Aggregate** = mean of each metric over the answered in-domain subset; report N
  per metric (they differ — faithfulness drops no-claim rows, relevance drops
  noncommittal rows differently).

**Tasks (built by the user):**
- [ ] `LLMClient` judge methods (`src/summarization/llm_client.py`) — mirror
      `generate_grounded_answer`'s flat-`parse` pattern: `decompose_claims(answer)
      -> list[str]`, `verify_claims(context, statements) -> list[bool/verdict]`,
      `generate_candidate_questions(answer, n=3) -> {questions, noncommittal}`.
      One flat Pydantic response model each, `temperature=0`.
- [ ] `src/evaluation/judge.py` — `faithfulness(answer, context, llm)` and
      `answer_relevance(query, answer, llm, embedder)`; orchestrate the judge
      calls + cosine; pure metric math (fractions, cosine) factored so it
      unit-tests against mocked llm/embedder. No DB/Chroma import.
- [ ] Judge cache (`src/evaluation/judge_cache.py` or inside `judge.py`) — JSON on
      disk under `output/eval/`, keyed `(query_id, metric, hash(answer+context))`;
      cache hit skips the LLM call. Re-runs free + deterministic.
- [ ] Extend `src/evaluation/harness.py` — `evaluate_generation()` → iterate
      answered in-domain rows: `answer_query()` for the answer (cached),
      `retrieve()` for the context, score both metrics, aggregate → a
      `GenerationReport` dataclass (reuse `_percentile`, the `EvalReport` style).
- [ ] Extend `evaluate.py` — add `--judge` (run generation eval), `--gen-sample N`
      (cap queries), reuse `--testset`/`--out`. Write
      `output/eval/generation_eval_<date>.md` + `.json`; print a token + estimated
      $ summary (gpt-4o-mini pricing).
- [ ] Report — faithfulness mean (N), answer-relevance mean (N), noncommittal
      count, worst-K offenders table, cost/token summary, and caveats (judge is
      itself an LLM → variance; single config; subset if `--gen-sample` used).
- [ ] Tests (`tests/test_evaluation.py` or `tests/test_judge.py`) — faithfulness
      fraction on mocked statements/verdicts; no-claims exclusion; relevance cosine
      on mocked embeddings; noncommittal → 0; cache hit avoids the second LLM call.

**Done when:** `python evaluate.py --judge` loads the committed test set, generates
+ scores answers for the answered in-domain queries, prints mean faithfulness and
answer-relevance (with per-metric N), reports a token/$ summary, caches judge
outputs so a re-run is free, and states the judge-variance + single-config caveats.
`judge.py` metric math is unit-tested.

**Watch-outs:**
- **Judge is itself an LLM** — faithfulness/relevance are estimates with their own
  noise; `temperature=0` + cache for replay, and say so in the report.
- **Don't double-charge** — cache the generated `answer` per `query_id` so
  faithfulness and relevance reuse one generation, not two.
- **No-claim / noncommittal rows** — exclude from the relevant mean (don't score
  1.0 or 0 silently); report the dropped count so means stay honest.
- **Context must match what produced the answer** — rebuild the numbered context
  from `retrieve(query, top_k)` at the **same `top_k`** `answer_query()` used, or
  faithfulness checks against the wrong evidence.
- **Read-only over the corpus** — judge cache + reports are the only writes; never
  touch `data/news.db` / `data/chroma/`.
- **Hold config at defaults** — top_k / chunk / embedding model fixed here;
  sweeping them is **Ship H**.

**Deferred:** multi-config sweep (chunk size, embedding model, top_k) + union-
pooling → **Ship H**; distance relevance-floor tuning → **Ship I**; audit of the
assistant-labeled testset half → **Ship I**.

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
