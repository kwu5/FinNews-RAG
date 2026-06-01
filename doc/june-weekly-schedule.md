# June Schedule — FinNews-RAG (weekly goals)

**Goal:** Finish the project (Ships C–I) by end of June.
**Cadence:** one ship per week, weekends used as catch-up buffer.
**Why weekly (not 3-day):** past slippage has been multi-week *gaps*, not slow
coding. Weekly checkpoints with a weekend rescue valve fit that pattern — a
slipped week is recoverable Saturday; a slipped 3-day ship just feels failed.

**Source of truth for ship detail:** `IMPLEMENTATION_PLAN.md` (deadlines live
here, not there). Per-ship working docs: `doc/ship-c-chunking.md`, etc.

## Link-posting plan (June 1)

The link goes live **June 1**, when only **Ship C** is done — the demo (Ship D)
does **not** exist yet. This is fine **only if the README matches reality**:

- README must say plainly what works vs. what's in progress, e.g.:
  > Working: ingestion, dedup, chunk-level retrieval.
  > In progress: cited Q&A (this week), evaluation harness (June).
- Do **not** let `PROJECT_B_README.md` promise a working RAG Q&A + eval harness
  the code lacks — a reviewer clicking June 1 would catch the gap, which reads
  worse than an honest "in progress."
- The link **improves over the month**: add the Streamlit screenshot the moment
  Ship D lands → the link becomes demo-backed within Week 1.

## Weekly goals

| When | Goal | Notes |
|---|---|---|
| **Weekend May 30–31** | **Ship C** — chunking + chunk-level ChromaDB | Prereq for everything; scoped in `doc/ship-c-chunking.md` |
| **Jun 1** | 📌 Post link (honest README) | Infrastructure visible; demo marked in-progress |
| **Week 1 · Jun 1–7** | **Ship D** — retriever + cited Q&A + Streamlit | The demo. Add screenshot to README when it lands → link becomes demo-backed |
| **Week 2 · Jun 8–14** | **Ship E** — labeled test set (50–100 pairs) | Schedule wildcard — slow manual labeling; gets a full week to itself |
| **Week 3 · Jun 15–21** | **Ships F + G** — eval harness (P/R + latency/cost, then RAGAS) | The tight week; weekend is the buffer here |
| **Week 4 · Jun 22–28** | **Ship H** — multi-config runner + written findings | The interview story: "measured X, chose Y because Z" |
| **Jun 29–30** | **Ship I** — polish + README finalize | Also absorbs slippage from the month |

## Risk & release valves

- **Tightest week:** Week 3 (two ships). Weekend buffer is reserved for it.
- **Biggest wildcard:** Ship E. Hand-labeling 50–100 pairs is manual, not
  coding. If it drags, pull queries straight from articles already in the DB to
  speed labeling.
- **Hard cap:** if behind, compress **Ship H** ("ship with defaults — chunk 256,
  top-k 5 — if config tuning is inconclusive," per the risk register). Never
  cut D or E.
- **Milestone over percentage:** after D = "working cited-answer demo" (the real,
  postable claim); after H = "evaluated across configs, settings backed by
  measured precision/recall + faithfulness" (the interview claim).
