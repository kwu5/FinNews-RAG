"""src/evaluation/harness.py — Ship F: score retrieve() against the test set.

Article-level retrieval metrics over the in-domain queries + out-of-domain
abstention, with per-query retrieval latency. Pure metric math lives in
`metrics.py`; this module owns the retrieving, the chunk->article dedup, the
timing, and the aggregation.

What it calls:
  - in-domain rows: `Retriever.retrieve()` ONLY (no LLM) -> P/R/hit/MRR + latency.
  - out-of-domain rows: `QAEngine.answer_query()` -> abstention. This is the only
    place the LLM is touched; OOD is a small flagged minority so the run stays cheap.

Read-only: never writes to the DB or ChromaDB.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Sequence

from src.evaluation import judge, metrics
from src.evaluation.testset import TestQuery

METRIC_NAMES = ("precision", "recall", "hit")


def dedup_to_articles(hits: Sequence[dict]) -> List[int]:
    """Chunk hits -> ordered unique article ids (keep the first/best-ranked per article)."""
    seen = set()
    ordered: List[int] = []
    for h in hits:
        aid = h["article_id"]
        if aid not in seen:
            seen.add(aid)
            ordered.append(aid)
    return ordered


@dataclass
class QueryResult:
    query_id: str
    type: str
    per_k: Dict[int, Dict[str, float]] = field(default_factory=dict)  # in-domain metrics by k
    mrr: Optional[float] = None                                       # in-domain, at deepest k
    latency_ms: Optional[float] = None                               # retrieval latency (in-domain)
    answered_from_context: Optional[bool] = None                     # out-of-domain
    abstention_correct: Optional[bool] = None                        # out-of-domain


@dataclass
class EvalReport:
    top_k: int
    k_values: List[int]
    n_in_domain: int
    n_out_of_domain: int
    per_k_means: Dict[int, Dict[str, float]]
    mean_mrr: float
    abstention_accuracy: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    meta: Dict = field(default_factory=dict)
    results: List[QueryResult] = field(default_factory=list)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty sample."""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(s)))
    return s[rank - 1]


def evaluate_retrieval(
    testset: List[TestQuery],
    retriever,
    qa_engine,
    top_k: int,
    k_values: Sequence[int],
) -> EvalReport:
    """Run the retrieval eval. `retriever` needs .retrieve(query, k); `qa_engine`
    needs .answer_query(query, k) -> object with .answered_from_context."""
    k_values = sorted(set(k_values))
    max_k = max([*k_values, top_k])

    # Warm the embedding model once so cold-start cost doesn't skew latency.
    if testset:
        retriever.retrieve(testset[0].query, 1)

    results: List[QueryResult] = []
    latencies: List[float] = []

    for q in testset:
        if q.is_out_of_domain:
            ga = qa_engine.answer_query(q.query, top_k)
            results.append(
                QueryResult(
                    query_id=q.query_id,
                    type=q.type,
                    answered_from_context=ga.answered_from_context,
                    abstention_correct=metrics.abstention_correct(ga.answered_from_context, True),
                )
            )
            continue

        t0 = time.perf_counter()
        hits = retriever.retrieve(q.query, max_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        # Slice CHUNKS to depth k, THEN dedup to articles — this mirrors what the
        # system actually surfaces at depth k (top-k chunks may share an article).
        per_k: Dict[int, Dict[str, float]] = {}
        for k in k_values:
            ranked = dedup_to_articles(hits[:k])
            per_k[k] = {
                "precision": metrics.precision(ranked, q.relevant_article_ids),
                "recall": metrics.recall(ranked, q.relevant_article_ids),
                "hit": metrics.hit_rate(ranked, q.relevant_article_ids),
            }
        ranked_full = dedup_to_articles(hits[:max_k])
        results.append(
            QueryResult(
                query_id=q.query_id,
                type=q.type,
                per_k=per_k,
                mrr=metrics.reciprocal_rank(ranked_full, q.relevant_article_ids),
                latency_ms=latency_ms,
            )
        )

    in_domain = [r for r in results if r.type == "in_domain"]
    ood = [r for r in results if r.type == "out_of_domain"]

    per_k_means: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        if in_domain:
            per_k_means[k] = {m: mean(r.per_k[k][m] for r in in_domain) for m in METRIC_NAMES}
        else:
            per_k_means[k] = {m: 0.0 for m in METRIC_NAMES}

    mean_mrr = mean(r.mrr for r in in_domain) if in_domain else 0.0
    abstention_accuracy = (
        mean(1.0 if r.abstention_correct else 0.0 for r in ood) if ood else None
    )

    return EvalReport(
        top_k=top_k,
        k_values=list(k_values),
        n_in_domain=len(in_domain),
        n_out_of_domain=len(ood),
        per_k_means=per_k_means,
        mean_mrr=mean_mrr,
        abstention_accuracy=abstention_accuracy,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        results=results,
    )


# --- Ship G: generation eval (faithfulness + answer-relevance) ----------------


@dataclass
class GenerationQueryResult:
    query_id: str
    answered: bool                              # answered_from_context
    faithfulness: Optional[float] = None        # None = answered but no factual claims
    answer_relevance: Optional[float] = None    # None only when not answered


@dataclass
class GenerationReport:
    top_k: int
    n_in_domain: int            # in-domain rows considered (after the sample cap)
    n_answered: int             # judged (answered_from_context = True)
    n_skipped_unanswered: int   # abstained -> nothing to judge
    mean_faithfulness: Optional[float]
    n_faithfulness: int         # answered rows with >=1 claim (faithfulness not None)
    mean_answer_relevance: Optional[float]
    n_answer_relevance: int     # answered rows (relevance always computed when answered)
    n_zero_relevance: int       # answered rows scoring 0.0 (noncommittal or off-topic)
    prompt_tokens: int
    completion_tokens: int
    est_cost_usd: float = 0.0   # filled by evaluate.py (pricing lives there)
    meta: Dict = field(default_factory=dict)
    results: List[GenerationQueryResult] = field(default_factory=list)


def _build_context(hits: Sequence[dict]) -> str:
    """Rebuild the numbered context block qa.py feeds the LLM, so faithfulness is
    checked against the SAME evidence that produced the answer (same top_k)."""
    return "".join(f"[{i}] {h['text']}\n\n" for i, h in enumerate(hits, start=1))


def evaluate_generation(
    testset: List[TestQuery],
    retriever,
    qa_engine,
    llm,
    embedder,
    top_k: int,
    cache,
    sample: Optional[int] = None,
) -> GenerationReport:
    """Score generated answers on faithfulness + answer-relevance over the
    in-domain queries that were actually answered. `cache` is a JudgeCache (or any
    object with key/get/set/save). Reads token totals off `llm` after the run.
    """
    in_domain = [q for q in testset if not q.is_out_of_domain]
    if sample is not None:
        in_domain = in_domain[:sample]

    results: List[GenerationQueryResult] = []
    for q in in_domain:
        ga = qa_engine.answer_query(q.query, top_k)
        if not ga.answered_from_context:
            results.append(GenerationQueryResult(query_id=q.query_id, answered=False))
            continue

        context = _build_context(retriever.retrieve(q.query, top_k))

        fkey = cache.key(q.query_id, "faithfulness", ga.answer, context)
        f = cache.get(fkey)
        if f is None:
            f = judge.faithfulness(ga.answer, context, llm)
            if f is not None:  # None (no claims) is left uncached by contract
                cache.set(fkey, f)

        rkey = cache.key(q.query_id, "answer_relevance", ga.answer, "")
        r = cache.get(rkey)
        if r is None:
            r = judge.answer_relevance(q.query, ga.answer, llm, embedder)
            cache.set(rkey, r)

        results.append(
            GenerationQueryResult(
                query_id=q.query_id, answered=True, faithfulness=f, answer_relevance=r
            )
        )

    cache.save()

    answered = [r for r in results if r.answered]
    f_vals = [r.faithfulness for r in answered if r.faithfulness is not None]
    r_vals = [r.answer_relevance for r in answered if r.answer_relevance is not None]

    return GenerationReport(
        top_k=top_k,
        n_in_domain=len(in_domain),
        n_answered=len(answered),
        n_skipped_unanswered=len(results) - len(answered),
        mean_faithfulness=mean(f_vals) if f_vals else None,
        n_faithfulness=len(f_vals),
        mean_answer_relevance=mean(r_vals) if r_vals else None,
        n_answer_relevance=len(r_vals),
        n_zero_relevance=sum(1 for v in r_vals if v == 0.0),
        prompt_tokens=getattr(llm, "total_prompt_tokens", 0),
        completion_tokens=getattr(llm, "total_completion_tokens", 0),
        results=results,
    )
