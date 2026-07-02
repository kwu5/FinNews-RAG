# Ship H — Config sweep findings

**Run:** 2026-06-30 · `python evaluate.py --sweep` · 5 configs / 3 index builds · $0.24
**Source:** `output/eval/sweep_eval_2026-06-30.{md,json}` (gitignored) · test set `eval/testset.jsonl` (q001–q093)
**Design:** see `doc/ship-h-config-sweep.md`. OFAT around the baseline (chunk 256 / top_k 5 / MiniLM); embedding model held fixed (Fork A); union-pooling deferred to Ship I (Fork B).

## Recommendation

**Keep the served defaults: chunk 256 / top_k 5 / MiniLM.**

No config beats the baseline by more than judge noise on the metric the call leans
on. Faithfulness across all five configs spans **0.938–0.981** — a ~0.04 band, with
the top four inside **0.024**. The nominal "winner" (chunk 512) leads the baseline by
**0.013 faithfulness**, at or below the ~0.01 (2 dp) tie threshold the ranking itself
declares noise — and it's *worse* on retrieval (hit-rate 0.795 vs 0.807). Crowning it
on that wobble would over-read the data. Per the risk register, when the sweep is
inconclusive we ship the defaults and say so. That's this case.

## Results

Sorted by the ranking key (faithfulness → answer-relevance → hit-rate/MRR):

| config | faithfulness | answer-rel | recall | hit-rate | MRR | precision | latency p50 |
|---|---|---|---|---|---|---|---|
| chunk512 / top_k5 | **0.981** | 0.814 | 0.770 | 0.795 | 0.729 | 0.232 | 19.0 ms |
| chunk256 / top_k5 ← baseline | 0.968 | 0.817 | 0.795 | 0.807 | 0.741 | 0.329 | 23.7 ms |
| chunk256 / top_k10 | 0.957 | 0.821 | 0.847 | 0.864 | 0.741 | 0.162 | 23.7 ms |
| chunk256 / top_k3 | 0.960 | 0.810 | 0.755 | 0.773 | 0.741 | 0.466 | 23.7 ms |
| chunk128 / top_k5 | 0.938 | **0.838** | 0.838 | 0.864 | **0.770** | 0.398 | 20.4 ms |

## What the axes actually show

The headline is "no decisive winner," but each OFAT arm shows a clean, monotonic
trade-off — those are the real findings, not the noisy faithfulness ranking.

### top_k (chunk 256, top_k ∈ {3, 5, 10}) — depth buys recall, costs precision

| top_k | recall | hit-rate | precision | faithfulness | answer-rel |
|---|---|---|---|---|---|
| 3 | 0.755 | 0.773 | 0.466 | 0.960 | 0.810 |
| 5 | 0.795 | 0.807 | 0.329 | 0.968 | 0.817 |
| 10 | 0.847 | 0.864 | 0.162 | 0.957 | 0.821 |

Going deeper monotonically lifts recall/hit-rate and monotonically tanks precision —
the textbook depth trade. Crucially, **the generation metrics barely move** (faithfulness
±0.01, answer-relevance +0.011 across a 3×-deeper retrieval). The LLM is robust to a
little extra context, so paying for top_k 10 buys retrieval recall the *answer* doesn't
cash in. top_k 5 is the sensible middle.

> Note: MRR is identical (0.741) across this arm. The three top_k configs share one
> physical chunk-256 index, and the harness reports MRR at the sweep's deepest k — so
> rank-of-first-relevant is the same regardless of the served top_k. MRR is not a
> top_k-discriminating metric in this setup.

### chunk_size (top_k 5, chunk ∈ {128, 256, 512}) — grounding vs. retrieval granularity

| chunk | faithfulness | answer-rel | recall | hit-rate | MRR |
|---|---|---|---|---|---|
| 128 | 0.938 | 0.838 | 0.838 | 0.864 | 0.770 |
| 256 | 0.968 | 0.817 | 0.795 | 0.807 | 0.741 |
| 512 | 0.981 | 0.814 | 0.770 | 0.795 | 0.729 |

A clean **inverse** relationship: as chunks grow, faithfulness rises (0.938 → 0.981)
while every retrieval metric falls (recall 0.838 → 0.770, hit-rate 0.864 → 0.795,
MRR 0.770 → 0.729). The interpretation:

- **Small chunks retrieve sharply but ground weakly** — tight 128-token passages match
  queries precisely (best hit-rate/MRR) but hand the LLM thin, fragmented context, so a
  few answer claims drift unsupported (lowest faithfulness). Best answer-relevance, too —
  granular passages keep the answer on-topic.
- **Large chunks ground well but retrieve bluntly** — 512-token passages give the LLM
  rich, self-contained context (highest faithfulness) but dilute the embedding, so the
  right passage ranks lower (worst retrieval).
- **256 sits at the knee** — second on faithfulness, mid-pack on retrieval, and it's the
  config the test set was actually pooled and labeled against.

The two halves pull in opposite directions and the spread on each is small, so neither
axis offers a free lunch over the baseline.

## Why the call leans on generation metrics, not retrieval

Retrieval P/R/MRR carries **pooling bias**: the test set was hand-labeled from the
*baseline* config's retrieval pool, so the baseline has home-field advantage and
non-baseline configs are scored against a relevant-set that never saw what they
surfaced. Faithfulness and answer-relevance judge the answer against its *own* retrieved
context and never touch the test-set labels, so they're immune. That's why the ranking
keys on them — and also why we don't read too much into chunk 128's strong hit-rate.
Union-pooling to remove the bias is scoped into Ship I (Fork B).

## Caveats

- **Within judge noise.** The faithfulness spread (0.04) is comparable to LLM-judge
  variance; treat the top-of-table ordering as a coin-flip, not a result.
- **OFAT, not a full grid.** One knob varied at a time around the baseline — interactions
  (e.g. chunk 512 only paying off at top_k 10) are unmeasured.
- **Pooling bias on retrieval** (above) — the reason the headline rests on generation
  metrics.
- **Single test set, 93 queries, MiniLM fixed.** Narrow slice; the embedding-model axis
  (the usual highest-leverage knob) was cut for cost and is the obvious next experiment.

## Carry-over to Ship I

- Union-pooling re-label so retrieval P/R is apples-to-apples across configs.
- Distance relevance-floor tuning (abstain when best distance too large) — orthogonal to
  the knobs swept here.
- The embedding-model axis (mpnet) remains the highest-value unrun experiment if revived.
