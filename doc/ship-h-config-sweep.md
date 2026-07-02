# Ship H — Multi-config comparison runner + written findings

**Status:** DONE 2026-06-30 — sweep ran ($0.24, 5 configs / 3 builds); findings in
`doc/ship-h-findings.md` (recommendation: keep defaults, results within judge noise).
Both forks resolved 2026-06-26 (Fork A = hold MiniLM fixed; Fork B = defer
union-pooling). Forks below kept for the reasoning record; the Decisions table is locked.
**Parent plan:** `IMPLEMENTATION_PLAN.md` (roadmap ship H)
**Predecessors:** Ship F (retrieval P/R + harness/CLI plumbing) ✅, Ship G
(faithfulness + answer-relevance + judge cache) ✅
**Schedule:** see `doc/june-weekly-schedule.md`.

## Goal

Run the eval harness across a small grid of retrieval/index configs and **write up
which one wins**. For each config run *both* metric halves — Ship F retrieval
P/R/MRR + latency, and Ship G faithfulness + answer-relevance + cost — collect them
into one comparison table, and name a recommended config in a findings doc.

This is the payoff of building the harness: turning "measurably this good"
(Ships F+G, single default config) into "measurably better than these
alternatives." That eval-backed config story is the README / interview claim.

## Why now

Ships F + G built and verified both metric halves at the **default** config
(chunk 256, top_k 5, MiniLM). Nothing yet varies the knobs the plan flagged as eval
axes. H is the last *eval* ship; I is polish / README / audit. H must follow G
because it re-runs G's generation eval once **per config**.

---

## Core concepts (read this if the forks didn't land)

### What "a config" is

A **config** is one setting of the knobs that affect retrieval quality:

| Knob | What it changes | Cost to change |
|---|---|---|
| `chunk_size` (+ `overlap`) | How big each indexed passage is. Smaller = more, tighter chunks; larger = fewer, broader chunks. | **Expensive** — must re-chunk + re-embed the whole corpus into a fresh index. |
| `embedding_model` | Which model turns text → vectors (MiniLM vs. mpnet). | **Expensive** — must re-embed the whole corpus; mpnet is a bigger, slower model. |
| `top_k` | How many chunks `retrieve()` returns per query. | **Free** — just a query-time number; the stored index doesn't change. |

So the **cost structure** is: changing chunk_size or embedding_model forces a
rebuild of the vector index; changing top_k does not. That single fact drives the
whole sweep design (build index once per (chunk, model) pair, loop top_k innermost
for free).

### OFAT vs. full grid

- **Full grid** = every combination of every knob. 3 chunk sizes × 2 models ×
  3 top_k = 18 configs, and the expensive part (chunk × model = 6 index builds)
  all has to happen. Explodes fast.
- **OFAT** = "one factor at a time." Start at the **baseline** (chunk 256, top_k 5,
  MiniLM) and vary **one** knob at a time, holding the others at baseline:
  - chunk_size ∈ {128, **256**, 512}, top_k=5, MiniLM  → 3 runs (3 index builds)
  - top_k ∈ {3, **5**, 10}, chunk=256, MiniLM          → 3 runs (0 new builds — same index)
  - (Fork A) model ∈ {**MiniLM**, mpnet}, chunk=256, top_k=5 → +1 build if included

OFAT shows each axis's *individual* effect on the metrics without paying for every
interaction. It can miss interactions (maybe 512-chunk only shines at top_k=10),
but it's the bounded, defensible choice for a portfolio project and matches the
risk-register hard-cap on Ship H time. **Decision (locked): OFAT, not full grid.**

### Index isolation (why canonical data stays safe)

Ships F + G are **read-only over the corpus** — they never write `data/chroma/` or
`data/news.db`. H must keep that promise even though it *builds new indexes*. So
each (chunk_size, embedding_model) config builds into a **throwaway** persist dir:

```
data/chroma_sweep/<config-slug>/   # gitignored, deleted/rebuilt freely
data/chroma/                       # canonical — NEVER touched by the sweep
```

The sweep reads articles from SQLite **read-only**, re-chunks + re-embeds into its
own throwaway dir, and points a `Retriever` at that dir. `data/chroma/` and
`data/news.db` are untouched.

---

## The two open forks (explained in full)

### FORK A — include a second embedding model, or hold MiniLM fixed? → **RESOLVED: hold MiniLM fixed (2026-06-26)**

**The choice:** is `embedding_model` one of the swept axes, or pinned at MiniLM?

**Why it's a real fork:** the embedding model is usually the *highest-leverage*
retrieval knob — but it's also the *most expensive* axis. Including mpnet doubles
the number of expensive index builds (3 chunk sizes → 6 builds), and mpnet
(~420 MB, ~5× MiniLM) is slower to embed the whole corpus and bigger on disk.

| | Hold MiniLM fixed (lean) | Include `all-mpnet-base-v2` |
|---|---|---|
| Index builds | 3 (chunk sizes only) | 6 (chunk × model) |
| Story | "Compared chunk size + top_k" | "Compared embedding models too" — strongest single addition |
| Time/disk risk | Low — under the hard-cap | High — the cuttable axis most likely to overrun |
| Reviewer Q "why MiniLM?" | Answer = original locked decision (local/free/fast) | Answer = eval evidence (a real P/R delta) |

**Lean:** hold MiniLM fixed. chunk_size + top_k already give a defensible OFAT
story, and the embedding axis is the explicit release valve in the plan if
time-boxed. Including mpnet is a *nice-to-have* that mostly buys a stronger
narrative, not a better product. **Open — your call.**

### FORK B — light union-pooling now, or defer to the Ship I audit? → **RESOLVED: defer to Ship I (2026-06-26)**

This one needs the terms first.

**How your test set was built (Ship E):** for each query you ran `retrieve()` to a
deep pool depth (~20–30), then **hand-labeled** those pooled candidates as relevant
/ not. So a query's ground-truth "relevant article" set only contains articles that
*your default config surfaced into the pool*.

**Pooling bias** (the problem): because the labels came from **one** config's pool,
that config has home-field advantage. Worked example —

> Config A (chunk 256, the one that did the pooling) and Config B (chunk 512) are
> both being compared. Config B correctly retrieves article #57, which is genuinely
> relevant to the query. But #57 was never in Config A's pool, so it was never
> shown to you, so it isn't in the labeled relevant set. The scorer sees #57 as
> **not relevant** → counts it a false positive → Config B's precision/recall look
> *worse* than they really are, purely because it surfaced something the labeling
> never considered.

So **retrieval recall is biased toward whatever config did the pooling.** A
different config can't win on retrieval metrics even if it's genuinely better.

**Union-pooling** (the standard fix): pool candidates from **all** configs you're
comparing, take the **union** of those candidate sets, and label *that* combined
set. Then every config's good retrievals had a chance to be labeled — no
home-field advantage. **"Light"** = a partial version: only label the *new*
candidates the other configs surface (you've already labeled the baseline's pool).

**Why generation metrics dodge this entirely:** faithfulness and answer-relevance
judge the **answer against its own retrieved context** — they never consult the
testset labels at all. So they are **immune** to pooling bias. *Only* retrieval
P/R/MRR is affected.

| | Defer to Ship I (lean) | Light union-pooling now |
|---|---|---|
| Headline config call based on | faithfulness/relevance (bias-free) | retrieval P/R now also trustworthy |
| Retrieval P/R across configs | reported with pooling-bias caveat | apples-to-apples within the sweep |
| Manual labeling in H | none | adds relabeling into H's tightest time-box |
| Duplication risk | none — already scoped into Ship I audit | overlaps the Ship I testset audit |

**Lean:** defer. The metric you'd actually base the "winner" on
(faithfulness/relevance) isn't biased, and union-pooling work is already on the
Ship I audit list — doing it in H risks doing it twice. Report retrieval P/R with
the standing caveat. **Open — your call.**

---

## Decisions

| Decision | Choice | Status |
|---|---|---|
| Sweep shape | OFAT around the default baseline, not a full grid | **Locked** |
| Axes + values | chunk_size ∈ {128, **256**, 512}; top_k ∈ {3, **5**, 10} | **Locked** |
| Embedding-model axis | **MiniLM fixed** (mpnet cut — expensive, nice-to-have only) | **Locked (Fork A, 2026-06-26)** |
| Index isolation | per-config throwaway `data/chroma_sweep/<slug>/`; canonical untouched | **Locked** |
| top_k handling | query-time only, no re-index; loop innermost | **Locked** |
| Pooling bias | lean on bias-free generation metrics; **defer union-pooling to Ship I**; report retrieval P/R with the standing caveat | **Locked (Fork B, 2026-06-26)** |
| Cost | each config pays a full generation-eval (~$0.05); OFAT grid ≈ well under $1 | **Locked** |

## Seams it consumes (reuse, do not fork)

- `evaluate_retrieval()`, `evaluate_generation()`, `EvalReport`,
  `GenerationReport`, `_percentile` — `src/evaluation/harness.py`.
- `JudgeCache` — `src/evaluation/judge_cache.py` (busts correctly when
  answer+context change per config, so each config re-judges — that's why each pays
  full generation-eval cost).
- `chunk_article` — `src/rag/chunker.py` (re-chunk per config).
- `VectorStore.add_chunks` + `Retriever` — pointed at a **per-config persist dir**.
- `EmbeddingGenerator` — `src/processing/embeddings.py`, **parameterized by model
  name** (only matters if Fork A includes mpnet).
- `load_testset` + the same `eval/testset.jsonl`, **held fixed** across all configs.

## Tasks

- [x] **Sweep config model** — `SweepConfig` + `build_grid` (OFAT, baseline emitted once).
- [x] **Per-config index builder** (`src/evaluation/sweep.py`) — reads SQLite read-only,
      chunks at the config's size, embeds with MiniLM, writes to
      `data/chroma_sweep/<slug>/`; resumable (skips rebuild when collection count > 0).
- [x] **Sweep runner** — `run_sweep`: groups by `index_slug`, builds once per index,
      varies top_k innermost; one `SweepRow` per config.
- [x] **CLI** — `evaluate.py --sweep`; writes `output/eval/sweep_eval_<date>.{md,json}`
      with the ranked comparison table + grid-total token/cost.
- [x] **Findings doc** (`doc/ship-h-findings.md`) — comparison table, recommendation
      (keep defaults — within judge noise), per-axis trade-offs, caveats.
- [x] **Tests** (`tests/test_sweep.py`) — grid shape / build-once-per-slug / ranking
      over mocked harness results. 10/10 green.

## Done when

`python evaluate.py --sweep` builds isolated per-config indexes, runs both eval
halves against the fixed test set, writes a comparison table ranking configs on
retrieval P/R/MRR + faithfulness + answer-relevance + latency + cost, and
`doc/ship-h-findings.md` names a recommended config with reasoning — all while
`data/chroma/` and `data/news.db` stay untouched. If results are inconclusive, ship
with the defaults (chunk 256, top_k 5, MiniLM) per the risk register and say so.

## Watch-outs

- **Never write to canonical `data/chroma/`** — every config builds into a
  throwaway dir; gitignore `data/chroma_sweep/`.
- **Hold the test set fixed** across all configs or the comparison isn't
  apples-to-apples.
- **Order the sweep to minimize re-indexing** — chunk/model change = expensive
  rebuild; top_k change = free; loop top_k innermost.
- **Embedding-model axis is the expensive, cuttable one** (Fork A) — if time-boxed,
  fix the embedder and sweep only chunk + top_k.
- **Pooling bias hits retrieval recall, not generation metrics** (Fork B) — lean on
  faithfulness/relevance for the config call; state the recall caveat.
- **Judge cache busts correctly across configs** (answer+context differ) — correct,
  but means each config pays full generation-eval cost; budget for it.

## Deferred to Ship I

- Distance relevance-floor tuning (abstain when best distance too large).
- Audit of the assistant-labeled testset half + q010-style pooling misses.
- Full union-pooling re-labeling if not done here (Fork B).
- Streamlit polish + README finalize.
