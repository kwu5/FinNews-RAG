# Ship G — Eval harness pt.2: faithfulness + answer-relevance

**Status:** Planned — detail drafted 2026-06-18 (not yet built)
**Parent plan:** `IMPLEMENTATION_PLAN.md` (roadmap ship G)
**Predecessors:** Ship D (`answer_query()` seam) ✅, Ship E (`eval/testset.jsonl`
ground truth) ✅, Ship F (retrieval P/R + harness/CLI plumbing) ✅
**Schedule:** see `doc/june-weekly-schedule.md` (Week 3, Jun 15–21 — Ships F + G).

## Goal

Score the **generated answer** (not just retrieval) with an LLM-judge and put
numbers on generation quality:

- **Faithfulness** — every claim in `answer` is supported by the retrieved
  context (the externalized, per-claim form of the old v2 "hallucination check").
- **Answer-relevance** — the answer actually addresses the query, not evasive or
  padded.

Report mean faithfulness / answer-relevance over the **answered in-domain**
queries (with per-metric N), plus a token/cost summary. Output a Markdown + JSON
report under `output/eval/`. This is the second half of the headline
differentiator and the first eval half that spends API money.

## Why now

Ship F measured retrieval but never judged the answer text. H sweeps configs and
needs both metric halves in place to compare, so G sits between them. It reuses
Ship F's harness/CLI plumbing and the existing `LLMClient` structured-output
pattern — extend, don't fork.

## Decisions (locked 2026-06-18)

| Decision | Choice | Rationale |
|---|---|---|
| Metric implementation | **Custom LLM-judge**, not the RAGAS library | RAGAS pulls in LangChain → conflicts with the locked "no orchestration framework" decision (plan line 25); we keep RAGAS's metric *definitions* and report under those names, computing them ourselves with `LLMClient` + MiniLM. More defensible in interview ("I implemented faithfulness, here's how") |
| Faithfulness method | Claim decomposition → per-claim support check vs. context | RAGAS definition: `supported_claims / total_claims` |
| Answer-relevance method | Generate N=3 reverse-questions from the answer → mean cosine to the original query (MiniLM); noncommittal answer → 0 | RAGAS definition; reuses the local embedder — the cosine step costs no API |
| Scope | **In-domain queries that were answered** (`answered_from_context=True`) only | OOD rows abstain (no answer to judge); already scored as abstention in Ship F |
| Cost control | Disk **cache** keyed `(query_id, metric, hash(answer+context))`; `--gen-sample N` cap; print token + $ estimate | Risk register: subset, cache judge outputs, budget per run. gpt-4o-mini makes the full set cents, but cache buys free + deterministic re-runs |
| Determinism | Judge calls at `temperature=0`; cache locks replay | Judge is itself an LLM — note its variance as a caveat |
| Mutation | Read-only over DB + ChromaDB (writes only the judge cache + report) | Eval must not alter the corpus it measures |

## Seams it consumes (do not modify these)

- **`qa.answer_query(query, top_k) -> GroundedAnswer`** (`src/rag/qa.py`) —
  `answer`, `citations`, `answered_from_context`. Generating an answer here costs
  an LLM call now (Ship F only invoked it for OOD).
- **`Retriever.retrieve(query, top_k) -> list[dict]`** (`src/rag/retriever.py`) —
  each hit carries `text`; rebuilds the numbered context block faithfulness checks
  against. Free/deterministic.
- **`LLMClient`** (`src/summarization/llm_client.py`) — already uses
  `beta.chat.completions.parse(..., temperature=0, response_format=Model)` with a
  flat Pydantic model (see `generate_grounded_answer` / `GroundedLLMResponse`).
  Add judge methods in the same shape.
- **`EmbeddingGenerator`** (`src/processing/embeddings.py`) —
  `normalize_embeddings=True` (per spec) so cosine = dot product. Confirm/reuse a
  cosine helper (`deduplicator.py` has `cosine_similarity`) rather than re-adding.
- **`eval/testset.jsonl`** in-domain rows; the Ship F harness (`EvalReport`,
  `_percentile`), CLI (`evaluate.py`), and loader (`src/evaluation/testset.py`).

## Metric definitions (as implemented)

- **faithfulness** = `supported_claims / total_claims`. Decompose `answer` into
  atomic statements (1 LLM call); verify all statements against the numbered
  context in **one batched** call (verdict + reason per statement). No factual
  claims (e.g. pure abstention text) → **excluded** from the mean (report the
  count), not scored 1.0.
- **answer-relevance** = `mean(cosine(q_i, original_query))` over N=3 questions an
  LLM generates *from the answer* (1 LLM call); embed with MiniLM. If the
  generator flags the answer **noncommittal** ("I don't know"-style) → score 0.
- **Aggregate** = mean of each metric over the answered in-domain subset; report N
  per metric — they differ, because faithfulness drops no-claim rows and relevance
  drops noncommittal rows on different criteria.

## Tasks (built by the user)

- [ ] **`LLMClient` judge methods** (`src/summarization/llm_client.py`) — mirror
      `generate_grounded_answer`'s flat-`parse` pattern: `decompose_claims(answer)
      -> list[str]`, `verify_claims(context, statements) -> list[bool/verdict]`,
      `generate_candidate_questions(answer, n=3) -> {questions, noncommittal}`.
      One flat Pydantic response model each, `temperature=0`.
- [ ] **`src/evaluation/judge.py`** — `faithfulness(answer, context, llm)` and
      `answer_relevance(query, answer, llm, embedder)`; orchestrate the judge calls
      + cosine; pure metric math (fractions, cosine) factored so it unit-tests
      against mocked llm/embedder. No DB/Chroma import.
- [ ] **Judge cache** (`src/evaluation/judge_cache.py` or inside `judge.py`) — JSON
      on disk under `output/eval/`, keyed `(query_id, metric, hash(answer+context))`;
      cache hit skips the LLM call. Re-runs free + deterministic.
- [ ] **Extend `src/evaluation/harness.py`** — `evaluate_generation()`: iterate
      answered in-domain rows; `answer_query()` for the answer (cached),
      `retrieve()` for the context, score both metrics, aggregate → a
      `GenerationReport` dataclass (reuse `_percentile`, the `EvalReport` style).
- [ ] **Extend `evaluate.py`** — add `--judge` (run generation eval),
      `--gen-sample N` (cap queries), reuse `--testset` / `--out`. Write
      `output/eval/generation_eval_<date>.md` + `.json`; print a token + estimated
      $ summary (gpt-4o-mini pricing).
- [ ] **Report** — faithfulness mean (N), answer-relevance mean (N), noncommittal
      count, worst-K offenders table, cost/token summary, and caveats (judge is
      itself an LLM → variance; single config; subset if `--gen-sample` used).
- [ ] **Tests** (`tests/test_evaluation.py` or `tests/test_judge.py`) —
      faithfulness fraction on mocked statements/verdicts; no-claims exclusion;
      relevance cosine on mocked embeddings; noncommittal → 0; cache hit avoids the
      second LLM call.

## Done when

`python evaluate.py --judge` loads the committed test set, generates + scores
answers for the answered in-domain queries, prints mean faithfulness and
answer-relevance (with per-metric N), reports a token/$ summary, caches judge
outputs so a re-run is free, and states the judge-variance + single-config caveats
in the output. `judge.py` metric math is covered by unit tests.

## Watch-outs

- **Judge is itself an LLM** — faithfulness/relevance are estimates with their own
  noise; `temperature=0` + cache for replay, and say so in the report.
- **Don't double-charge** — cache the generated `answer` per `query_id` so
  faithfulness and relevance reuse one generation, not two.
- **No-claim / noncommittal rows** — exclude from the relevant mean (don't score
  1.0 or 0 silently); report the dropped count so means stay honest.
- **Context must match what produced the answer** — rebuild the numbered context
  from `retrieve(query, top_k)` at the **same `top_k`** `answer_query()` used, or
  faithfulness checks against the wrong evidence. (Note `answer_query()` retrieves
  internally; re-retrieving at the same top_k reproduces that context exactly.)
- **Read-only over the corpus** — judge cache + reports are the only writes; never
  touch `data/news.db` / `data/chroma/`.
- **Hold config at defaults** — top_k / chunk / embedding model fixed here;
  sweeping them is **Ship H**.

## Deferred to later ships

- Multi-config sweep (chunk size, embedding model, top_k) + union-pooling across
  configs → **Ship H**.
- Distance relevance-floor tuning (abstain when best distance is too large) →
  **Ship I**.
- Audit of the assistant-labeled testset half → **Ship I**.
