# FinNews-RAG — Project Guide (current state)

**Purpose:** the single doc to read after time away — what the project is, how it's
built, how to run every surface, what's measured, and what's left. Reflects the code
as it actually exists at **Ship H committed (HEAD `474c7d1`)**.

- **Direction / roadmap:** `IMPLEMENTATION_PLAN.md`
- **Schedule:** `doc/june-weekly-schedule.md`
- **Per-ship working docs:** `doc/ship-*.md`
- **Superseded:** `doc/project-overview.md` is frozen at *mid–Ship C* (says RAGAS,
  lists D–I as planned) — this guide replaces it.

---

## 1. What this project is

It began as a **daily financial-news summarizer** (ingest → dedupe → LLM briefing →
Markdown/HTML report) and has been evolved **in place** into **FinNews-RAG**: a
retrieval-augmented Q&A system over the same news corpus, whose headline feature is a
**self-evaluation harness** — retrieval precision/recall, answer faithfulness &
relevance, latency/cost, measured across configurations.

Two product surfaces on one ingestion/storage backbone:

| Surface | Status | What it does |
|---|---|---|
| **RAG Q&A + eval** | ✅ Working (the headline) | Chunk → embed → retrieve top-k → grounded cited answer → measure quality |
| **Daily briefing** | ✅ Working (inherited MVP, secondary) | Fetch → clean → dedupe → summarize → dated report |

**Why the pivot:** a daily-summary report reads as a generic LLM demo. RAG Q&A + a
real evaluation harness is the defensible, interview-worthy angle. The product model
shifts from *push* (scheduled report) to *pull* (user query → grounded cited answer).
The briefing path summarizes in-memory article dicts directly via the LLM — it does
**not** read ChromaDB — so the chunk-level RAG index and the briefing coexist cleanly.

---

## 2. Status at a glance

| Ship | Focus | Status |
|---|---|---|
| A | RSS-first ingestion + full-text extraction + source health | ✅ committed |
| B | Dedup stages 1–2 (URL canon + content hash); DB wipe/recreate | ✅ committed |
| C | Chunking + chunk-level ChromaDB | ✅ committed |
| D | Retriever + grounded cited Q&A + Streamlit skeleton | ✅ committed |
| E | Labeled test set (`eval/testset.jsonl`, q001–q093) | ✅ committed |
| F | Eval harness pt.1 — retrieval P/R/MRR + latency | ✅ committed |
| G | Eval harness pt.2 — faithfulness + answer-relevance (LLM-judge) | ✅ committed |
| H | Multi-config OFAT sweep + written findings | ✅ committed (`474c7d1`) |
| **I** | **Streamlit polish + README finalize; testset audit; stretch: signals** | **🚧 in progress** |

**Ship I remaining (this is all that's left):**
1. **README finalize** — `README.md` is still the *old MVP text*; the real story is
   in `PROJECT_B_README.md` (which has stale claims — see §9). Decision made: fold the
   corrected FinNews-RAG content into `README.md`, retire `PROJECT_B_README.md`.
2. **Streamlit polish** + screenshot — see `doc/ship-i-streamlit-polish.md`.
3. **Deferred / documented as future work:** audit the assistant-labeled half of the
   test set + q010-style pooling misses; distance relevance-floor abstention;
   union-pooling re-label; stretch: structured signal extraction.

---

## 3. Architecture

```
                 ┌─────────────────────────────────────────────┐
  entry points   │ main.py (--mode api|run-once|scheduler)      │
                 │ app.py (streamlit)   evaluate.py (eval CLI)  │
                 └───────┬───────────────────┬─────────────┬────┘
                         │                   │             │
             ┌───────────▼──────┐   ┌────────▼───────┐  ┌──▼──────────────┐
   faces     │ api/main.py      │   │ rag/qa.py      │  │ evaluation/      │
             │ FastAPI (briefing)│  │ QAEngine       │  │ harness+judge+   │
             │ scheduler/jobs.py │  │ answer_query() │  │ sweep            │
             └───────────┬──────┘   └────────┬───────┘  └──┬──────────────┘
                         │                   │             │
                 ┌───────▼──────┐    ┌────────▼───────┐     │
                 │ pipeline.py  │    │ rag/retriever  │◄────┘
                 │ run_pipeline │    │ retrieve()     │
                 └───────┬──────┘    └────────┬───────┘
        ┌──────────┬─────┼──────────┬─────────┴──────┐
        ▼          ▼     ▼          ▼                ▼
    ingestion  processing storage  summarization   rag/chunker
    RSS + WNA  clean/dedup SQLite +  LLM + report   token-window
               /embeddings ChromaDB                 chunking
```

Key idea: **`pipeline.py` is the single orchestrator** for the ingest→index→brief
flow; both API and scheduler call `run_pipeline()`. The RAG path
(`retriever` → `qa`) and the eval path (`evaluation/*`) read the same ChromaDB index
but run independently of the pipeline. Heavy objects (embedding model, DB, vector
store, LLM client) are instantiated once per process, not per query.

---

## 4. Repo map (what each file does)

```
main.py                     Entry point — argparse → api | run-once | scheduler
app.py                      Streamlit RAG demo (Ship D skeleton; polish = Ship I)
evaluate.py                 Eval CLI — retrieval / --judge / --sweep

src/
├── config.py               Pydantic Settings — env-driven (keys, paths, chunk/top_k)
├── pipeline.py             run_pipeline(): full ingest→dedupe→index→brief→export
│
├── ingestion/
│   ├── rss_reader.py       PRIMARY. Parses config/feeds.yaml, normalizes, tracks health
│   ├── extractor.py        Full-text fallback: trafilatura → newspaper3k → readability → rss-only
│   ├── world_news_api.py   FALLBACK. Used only when RSS yield < RSS_YIELD_THRESHOLD (15)
│   └── search_news_sample.py  (sample/scratch)
│
├── processing/
│   ├── cleaner.py          TextCleaner — strip URLs/emails/ads, normalize, spaCy NER
│   ├── url_canon.py        canonicalize_url() — dedup key #1
│   ├── content_hash.py     compute_content_hash() — dedup key #2 (after cleaning)
│   ├── deduplicator.py     Deduplicator — embedding cosine-similarity dedup (stage 3)
│   └── embeddings.py       EmbeddingGenerator — wraps all-MiniLM-L6-v2 (normalize=True)
│
├── storage/
│   ├── database.py         SQLAlchemy — Article + DailyReport models, CRUD, get_all_articles()
│   └── vector_store.py     ChromaDB wrapper — add_chunks(), search_similar()
│
├── rag/                    ← the RAG path
│   ├── chunker.py          chunk_article() — token-windowed passage splitting (~256 tok, ~15% overlap)
│   ├── retriever.py        Retriever.retrieve(query, top_k) → flattened hit dicts
│   └── qa.py               QAEngine.answer_query() → GroundedAnswer (Citation, answered_from_context)
│
├── summarization/
│   ├── llm_client.py       LLMClient — grounded answer + judge methods + briefing + sentiment
│   └── report_generator.py ReportGenerator — Markdown + Jinja2 HTML export
│
├── evaluation/             ← the headline differentiator
│   ├── testset.py          load_testset() → typed TestQuery rows; integrity asserts
│   ├── metrics.py          pure fns: precision_at_k, recall_at_k, mrr, abstention
│   ├── harness.py          evaluate_retrieval() + evaluate_generation() → EvalReport/GenerationReport
│   ├── judge.py            faithfulness() + answer_relevance() (custom RAGAS-definition metrics)
│   ├── judge_cache.py      JSON disk cache keyed (query_id, metric, hash(answer+context))
│   └── sweep.py            Ship H — build_grid / build_index / run_sweep / rank_rows (OFAT)
│
├── api/main.py             FastAPI: / · /health · /generate · /report/{date} · /reports
└── scheduler/jobs.py       APScheduler — daily briefing at DAILY_RUN_HOUR

eval/testset.jsonl          Hand-labeled ground truth, q001–q093 (committed)
config/feeds.yaml           RSS feed list + per-feed config
tests/                      test_chunker, test_retriever, test_qa, test_judge,
                            test_evaluation, test_sweep, test_ingestion, test_processing,
                            test_summarization + smoke_ship_{b,c,d}.py
```

---

## 5. Data model

### SQLite (`data/news.db`) — source of record for articles
```
articles
  id              INTEGER PK autoincrement   ← becomes the chunk article_id
  title, description, content
  url             UNIQUE
  canonical_url   UNIQUE                     dedup key #1
  content_hash    UNIQUE                     dedup key #2
  source, published_at (UTC ISO 8601), fetched_at
  extraction_method   trafilatura|newspaper3k|readability|rss-only
  processed       bool  — included in a daily briefing
  indexed         bool  — chunked + embedded into ChromaDB (gates re-index)

daily_reports
  id, report_date UNIQUE, content, article_count, created_at
```

### ChromaDB (`data/chroma/`) — chunk-level vector index
- one entry per **chunk**, id `"{article_id}:{chunk_index}"`
- metadata: `article_id, chunk_index, title, source, url, published_at`
- document: chunk text
- This is what `retrieve()` and the eval harness query. Article-level embeddings from
  the MVP are gone. Eval maps a retrieved chunk → its `article_id` for scoring.

---

## 6. How to run everything

**Setup**
```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows (POSIX: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env                   # add NEWS_API_KEY + OPENAI_API_KEY
```

**Ingest + index (populates SQLite + ChromaDB)**
```bash
python main.py --mode run-once         # one synchronous pipeline run (fetch→dedupe→index→brief)
python main.py --mode api              # + FastAPI server (:8000) with daily scheduler
python main.py --mode scheduler        # scheduler only, blocking loop
```

**RAG demo**
```bash
streamlit run app.py                   # grounded, source-cited Q&A over the chunk index
```

**Evaluation harness**
```bash
python evaluate.py                     # retrieval P/R/MRR + latency at served top_k, plus k-sweep
python evaluate.py --judge             # generation eval: faithfulness + answer-relevance (LLM-judge)
python evaluate.py --sweep             # Ship H: OFAT multi-config sweep (~$0.24, throwaway indexes)
#   flags: --top-k, --k-sweep "1,3,5,10", --gen-sample N, --testset PATH, --out DIR
```
Reports write to `output/eval/` (gitignored). The judge cache makes warm re-runs of
`--judge` nearly free and deterministic.

**Tests**
```bash
pytest                                 # unit tests (chunker, retriever, qa, judge, sweep, ...)
```

---

## 7. The RAG path (retrieve → ground → cite)

`QAEngine.answer_query(query, top_k)`:
1. `retriever.retrieve(query, top_k)` → embed query, `search_similar`, flatten
   ChromaDB's nested result to hit dicts. **Empty hits → short-circuit**, no LLM call,
   `answered_from_context=False`.
2. Build a **numbered context block** (`[1] …`, `[2] …`) from the hit texts.
3. `llm.generate_grounded_answer(query, context)` — structured output,
   `temperature=0`, "answer only from context" prompt; returns `answer`,
   `used_markers`, `answered_from_context`.
4. **Citations are built from the retrieved hits, not the LLM.** The model picks which
   source *numbers* it used; we map each number back to the authoritative
   `chunk_id / article_id / url`. Out-of-range markers are dropped. → the model can
   mis-number a citation but cannot fabricate its target.

---

## 8. The evaluation harness (the headline)

Scored against `eval/testset.jsonl` — 93 hand-labeled queries (q001–q093), relevance
labeled at the **article** level (a retrieved chunk counts if its `article_id` is
labeled relevant). Out-of-domain rows (empty relevant set) are scored separately as
**abstention**, never folded into P/R.

**Retrieval (Ship F, custom, ~free):** precision / recall / hit-rate / MRR at the
article level, per-k; latency p50/p95 (embed + query). Cost N/A (local MiniLM + local
ChromaDB).

**Generation (Ship G, custom LLM-judge — implements the RAGAS *definitions*, no RAGAS
/ LangChain):**
- **Faithfulness** = `supported_claims / total_claims`. Decompose the answer into
  atomic claims (1 LLM call), verify each against the numbered context (1 batched
  call). No-claim rows excluded from the mean.
- **Answer-relevance** = mean cosine(original query, N=3 reverse-questions generated
  from the answer), embedded with local MiniLM. Noncommittal answers → 0.
- Judge = gpt-4o-mini @ `temperature=0`; cached on disk → free, deterministic re-runs.

**Headline numbers (defaults: chunk 256 / top_k 5 / MiniLM):**
- faithfulness **0.965** (n=78), answer-relevance **0.818** (n=78) — Ship G, ~$0.05/cold run
- retrieval: recall 0.795, hit-rate 0.807, MRR 0.741, precision 0.329, latency p50 ~24 ms

**Multi-config sweep (Ship H, OFAT, 5 configs, $0.24):** no config beats the baseline
by more than judge noise on faithfulness (**0.938–0.981** band). **Recommendation:
keep the defaults**, evidence-based. Clean per-axis trades found:
- **top_k↑** → recall↑ / precision↓, generation metrics flat (LLM robust to extra context)
- **chunk_size↑** → faithfulness↑ but retrieval↓ (inverse; 256 sits at the knee)

Full write-up: `doc/ship-h-findings.md`. **Retrieval P/R carries pooling bias** (labels
were pooled from the baseline config), so the winner call leans on the bias-free
generation metrics; union-pooling to fix it is deferred to the Ship I audit.

---

## 9. Key design decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Product | RAG Q&A + eval = headline; daily briefing = secondary | Defensible portfolio angle; pull over push |
| Scope | Evolve in place (same repo) | Pipeline already works — reuse |
| Orchestration | **None** — raw OpenAI / ChromaDB / sentence-transformers | Shows internals; no LangChain/LlamaIndex |
| Eval metrics | **Custom** implementation of RAGAS *definitions* + custom retrieval/latency | Keeps the no-framework decision; you built the internals |
| Embeddings | `all-MiniLM-L6-v2` (256-tok cap → 256-tok chunks) | Local, free, fast; also an eval axis |
| Generation LLM | OpenAI `gpt-4o-mini`, `temperature=0` | Cost-effective, reproducible |
| RSS strategy | RSS-first, World News API as fallback | Source diversity drives retrieval quality |
| Dedup | 3 stages: URL canon + content hash + embedding similarity | Prevents duplicate chunks polluting retrieval/eval |
| Test set | 93 pairs, article-level labels, hand-verified | Ground truth for retrieval metrics |
| Sweep shape | OFAT around the baseline, not full grid | Bounded cost; still shows each axis's effect |

> ⚠ **Doc/code mismatch to fix in README finalize:** `PROJECT_B_README.md` still
> claims **RAGAS** for faithfulness/answer-relevance (it's custom), lists **signal
> extraction** as a current capability (not built), and shows `--mode ingest` (not a
> real mode). Correct these when folding it into `README.md`.

---

## 10. Known limitations / caveats

- **LLM-judge is itself an LLM** — faithfulness/relevance are estimates with their own
  variance (the 0.04 sweep band ≈ judge noise). `temperature=0` + cache for replay.
- **Pooling bias on retrieval recall** — labels pooled from the baseline config; other
  configs are scored against a relevant-set that never saw what they surfaced.
  Generation metrics are immune. Union-pooling deferred to Ship I.
- **Small test set** — 93 queries, one embedding model — direction, not statistical
  certainty. The embedding-model axis (mpnet) is the highest-value unrun experiment.
- **Assistant-labeled testset half** (Ship E time-box) is flagged for the Ship I audit.
- **~10% full-text extraction failures** (trafilatura/newspaper3k) — those articles are
  dropped, not partially indexed.
- **OFAT, not a full grid** — cross-knob interactions unmeasured.

---

## 11. Doc map

| Doc | What it holds |
|---|---|
| `IMPLEMENTATION_PLAN.md` | Roadmap, locked decisions, per-ship detail sections |
| `doc/june-weekly-schedule.md` | When (deadlines) — separate from the plan's *what* |
| `doc/ship-c … ship-g *.md` | Per-ship working docs (design + watch-outs) |
| `doc/ship-h-config-sweep.md` | Ship H design (OFAT, forks, index isolation) |
| `doc/ship-h-findings.md` | Ship H results + recommendation (keep defaults) |
| `doc/ship-i-streamlit-polish.md` | Streamlit polish build guide (this session) |
| `doc/PROJECT_GUIDE.md` | ← you are here — current-state comprehensive guide |
| `PROJECT_B_README.md` | Portfolio README draft (to be finalized into README.md) |
