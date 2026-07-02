"""evaluate.py — Ship F entry point: retrieval precision/recall + latency.

Loads the committed test set (eval/testset.jsonl), scores `Retriever.retrieve()`
against it at the served top-k plus a k-sweep, reports out-of-domain abstention
accuracy separately, and writes a Markdown + JSON report under output/eval/.

Retrieval is local (MiniLM + ChromaDB) so it costs nothing; the only LLM calls are
the out-of-domain abstention checks (a handful). Read-only over the corpus.

Run:  python evaluate.py [--top-k N] [--k-sweep 1,3,5,10] [--testset PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime

from src.config import Settings
from src.evaluation.harness import (
    EvalReport,
    GenerationReport,
    evaluate_generation,
    evaluate_retrieval,
)
from src.evaluation.judge_cache import JudgeCache
from src.evaluation.sweep import build_grid, rank_rows, run_sweep
from src.evaluation.testset import TESTSET_PATH, load_testset
from src.processing.embeddings import EmbeddingGenerator
from src.rag.qa import QAEngine
from src.rag.retriever import Retriever
from src.storage.database import Database
from src.storage.vector_store import VectorStore
from src.summarization.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# gpt-4o-mini list price (USD per 1M tokens). VERIFY against current OpenAI pricing.
GPT_4O_MINI_INPUT_PER_1M = 0.15
GPT_4O_MINI_OUTPUT_PER_1M = 0.60


def _est_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1e6 * GPT_4O_MINI_INPUT_PER_1M
        + completion_tokens / 1e6 * GPT_4O_MINI_OUTPUT_PER_1M
    )


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def build_markdown(report: EvalReport) -> str:
    m = report.meta
    lines = [
        f"# Retrieval eval — {m.get('timestamp', '')}",
        "",
        f"**Test set:** `{m.get('testset_path')}` "
        f"({report.n_in_domain} in-domain, {report.n_out_of_domain} out-of-domain)  ",
        f"**Served top_k:** {report.top_k}  ·  **k-sweep:** {report.k_values}  ",
        f"**Embedding:** `{m.get('embedding_model')}`  ·  "
        f"**Chunk:** {m.get('chunk_size')}/{m.get('chunk_overlap')} tok  ·  "
        f"retrieval cost ≈ $0 (local); LLM calls: {report.n_out_of_domain} (OOD abstention only)",
        "",
        "## Retrieval quality (in-domain, article-level)",
        "",
        "| k | precision | recall | hit-rate |",
        "|---|-----------|--------|----------|",
    ]
    for k in report.k_values:
        row = report.per_k_means[k]
        marker = "  ← served" if k == report.top_k else ""
        lines.append(
            f"| {k} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['hit'])} |{marker}"
        )
    lines += [
        "",
        f"**MRR** (first relevant article, depth {max([*report.k_values, report.top_k])}): "
        f"{_fmt(report.mean_mrr)}",
        "",
        "## Out-of-domain abstention",
        "",
    ]
    if report.abstention_accuracy is None:
        lines.append("_No out-of-domain queries in the set._")
    else:
        n_correct = sum(1 for r in report.results if r.abstention_correct)
        lines.append(
            f"**{_fmt(report.abstention_accuracy)}** "
            f"({n_correct}/{report.n_out_of_domain} abstained correctly — "
            f"`answered_from_context=False` on off-topic queries)"
        )
    lines += [
        "",
        "## Latency (retrieval path, per query)",
        "",
        f"p50 {_fmt(report.latency_p50_ms, 1)} ms · p95 {_fmt(report.latency_p95_ms, 1)} ms  ",
        "_Embed query + ChromaDB search; ~independent of k. Excludes model cold-start._",
        "",
        "## Caveats",
        "",
        "- **Recall is capped by Ship E pooling.** Labels were pooled from this same "
        "retriever, and ~7 in-domain seeds were never surfaced (`retrieve()` missed them), "
        "so low recall is partly a labeling ceiling, not purely a retriever failure. "
        "Flagged for the Ship I audit.",
        "- **precision@k denominator = distinct articles surfaced by the top-k chunks** "
        "(not k). Most in-domain queries have a single relevant article, so **hit-rate / "
        "recall / MRR** are the meaningful headline; precision runs low by construction.",
        "- Single config (top_k, chunk size, embedding model held at defaults). "
        "Sweeping those is Ship H.",
        "",
    ]
    return "\n".join(lines)


def build_generation_markdown(report: GenerationReport) -> str:
    m = report.meta
    lines = [
        f"# Generation eval (faithfulness + answer-relevance) — {m.get('timestamp', '')}",
        "",
        f"**Test set:** `{m.get('testset_path')}` "
        f"(in-domain answered: {report.n_answered}/{report.n_in_domain}; "
        f"{report.n_skipped_unanswered} abstained, not judged)  ",
        f"**Served top_k:** {report.top_k}  ·  **LLM judge:** `{m.get('llm_model')}`  ·  "
        f"**Embedding:** `{m.get('embedding_model')}`",
        "",
        "## Generation quality (in-domain, answered)",
        "",
        "| metric | mean | n |",
        "|--------|------|---|",
        f"| faithfulness | {_fmt(report.mean_faithfulness)} | {report.n_faithfulness} |",
        f"| answer-relevance | {_fmt(report.mean_answer_relevance)} | {report.n_answer_relevance} |",
        "",
        f"_faithfulness n < answered ({report.n_answered}) where an answer had no "
        f"verifiable claims (excluded, not scored). {report.n_zero_relevance} answer(s) "
        f"scored 0 relevance (noncommittal or off-topic)._",
        "",
        "## Cost",
        "",
        f"Tokens: {report.prompt_tokens} in / {report.completion_tokens} out  ·  "
        f"est. **${report.est_cost_usd:.4f}** (gpt-4o-mini list price; generation + judge calls)  ",
        "_A re-run hits the judge cache, so repeated evals add ≈ only the answer-generation cost._",
        "",
        "## Caveats",
        "",
        "- **The judge is itself an LLM** — faithfulness/answer-relevance are estimates "
        "with their own noise. Calls are `temperature=0` and cached for replay, but "
        "treat small differences as noise.",
        "- **Single config** (top_k, chunk size, embedding model, judge model at "
        "defaults). Sweeping those is Ship H.",
    ]
    if m.get("sample") is not None:
        lines.append(
            f"- **Subset run** — `--gen-sample {m['sample']}`; not the full in-domain set."
        )
    lines.append("")
    return "\n".join(lines)


def _build_components(settings: Settings):
    """Build the RAG components directly (not via pipeline, to avoid ingestion
    side-effects). Returns (embedder, retriever, qa_engine, llm) or None if the
    index is empty."""
    embedder = EmbeddingGenerator(settings)
    vstore = VectorStore(settings)
    if vstore.financial_news_collection.count() == 0:
        logger.error("ChromaDB is empty — populate the index first (run the pipeline). Aborting.")
        return None
    llm = LLMClient(settings)
    retriever = Retriever(embedder, vstore)
    qa_engine = QAEngine(retriever, llm)
    return embedder, retriever, qa_engine, llm


def _run_retrieval(args, settings, testset, out_dir, components) -> None:
    embedder, retriever, qa_engine, _llm = components
    top_k = args.top_k if args.top_k is not None else settings.RETRIEVAL_TOP_K
    k_values = [int(x) for x in args.k_sweep.split(",") if x.strip()]

    logger.info("Evaluating %d queries (top_k=%d, k-sweep=%s)…", len(testset), top_k, k_values)
    report = evaluate_retrieval(testset, retriever, qa_engine, top_k, k_values)
    report.meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "testset_path": args.testset,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE_TOKENS,
        "chunk_overlap": settings.CHUNK_OVERLAP_TOKENS,
    }
    stamp = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(out_dir, f"retrieval_eval_{stamp}.md")
    json_path = os.path.join(out_dir, f"retrieval_eval_{stamp}.json")
    markdown = build_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, ensure_ascii=False, indent=2)
    print("\n" + markdown)
    print(f"\nWrote {md_path} and {json_path}")


def _run_generation(args, settings, testset, out_dir, components) -> None:
    embedder, retriever, qa_engine, llm = components
    top_k = args.top_k if args.top_k is not None else settings.RETRIEVAL_TOP_K
    cache = JudgeCache(os.path.join(out_dir, "judge_cache.json"))

    logger.info(
        "Judging generation on in-domain queries (top_k=%d, sample=%s)…",
        top_k, args.gen_sample,
    )
    report = evaluate_generation(
        testset, retriever, qa_engine, llm, embedder, top_k, cache, sample=args.gen_sample
    )
    report.est_cost_usd = _est_cost(report.prompt_tokens, report.completion_tokens)
    report.meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "testset_path": args.testset,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "sample": args.gen_sample,
    }
    stamp = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(out_dir, f"generation_eval_{stamp}.md")
    json_path = os.path.join(out_dir, f"generation_eval_{stamp}.json")
    markdown = build_generation_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(report), f, ensure_ascii=False, indent=2)
    print("\n" + markdown)
    print(f"\nWrote {md_path} and {json_path}")


# --- Ship H: multi-config sweep -----------------------------------------------


def _sweep_row_scalars(row) -> dict:
    """JSON-friendly view of one SweepRow: the config + the pulled-out scalars.
    Deliberately drops the nested EvalReport/GenerationReport — they're large,
    per-query, and duplicated across configs sharing an index."""
    return {
        "label": row.config.label,
        "index_slug": row.config.index_slug,
        "chunk_size": row.config.chunk_size,
        "overlap": row.config.overlap,
        "embedding_model": row.config.embedding_model,
        "top_k": row.config.top_k,
        "precision": row.precision,
        "recall": row.recall,
        "hit_rate": row.hit_rate,
        "mrr": row.mrr,
        "faithfulness": row.faithfulness,
        "answer_relevance": row.answer_relevance,
        "latency_p50_ms": row.latency_p50_ms,
    }


def build_sweep_markdown(ranked, meta) -> str:
    winner = ranked[0] if ranked else None
    lines = [
        f"# Config sweep (Ship H) — {meta.get('timestamp', '')}",
        "",
        f"**Test set:** `{meta.get('testset_path')}` "
        f"({meta.get('n_configs')} configs over {meta.get('n_index_builds')} index build(s))  ",
        f"**Embedding:** `{meta.get('embedding_model')}` (fixed — Fork A)  ·  "
        f"**LLM judge:** `{meta.get('llm_model')}`  ",
        "**Sweep:** OFAT around baseline chunk256 / top_k5 — one knob varied at a time.",
        "",
        "## Comparison (ranked best → worst)",
        "",
        "Ranked on the bias-free generation metrics (faithfulness, then answer-relevance), "
        "with hit-rate / MRR as tie-breakers — see caveats.",
        "",
        "| rank | config | faithfulness | answer-rel | hit-rate | MRR | precision | latency p50 (ms) |",
        "|------|--------|--------------|------------|----------|-----|-----------|------------------|",
    ]
    for i, row in enumerate(ranked, start=1):
        is_baseline = row.config.chunk_size == 256 and row.config.top_k == 5
        marks = []
        if i == 1:
            marks.append("winner")
        if is_baseline:
            marks.append("baseline")
        suffix = f"  ← {', '.join(marks)}" if marks else ""
        lines.append(
            f"| {i} | {row.config.label} | {_fmt(row.faithfulness)} | "
            f"{_fmt(row.answer_relevance)} | {_fmt(row.hit_rate)} | {_fmt(row.mrr)} | "
            f"{_fmt(row.precision)} | {_fmt(row.latency_p50_ms, 1)} |{suffix}"
        )
    lines += ["", "## Recommended config", ""]
    if winner is not None:
        lines.append(
            f"**{winner.config.label}** — faithfulness {_fmt(winner.faithfulness)}, "
            f"answer-relevance {_fmt(winner.answer_relevance)}, hit-rate {_fmt(winner.hit_rate)}. "
            "Confirm against the written findings (`doc/ship-h-findings.md`) before adopting; "
            "if the top configs are within judge noise, keep the defaults per the risk register."
        )
    else:
        lines.append("_No configs ran._")
    lines += [
        "",
        "## Cost (whole grid)",
        "",
        f"Tokens: {meta.get('total_prompt_tokens')} in / {meta.get('total_completion_tokens')} out  ·  "
        f"est. **${meta.get('total_cost_usd', 0.0):.4f}** total "
        "(gpt-4o-mini list price; answer-generation + judge across every config)  ",
        "_Judge calls are cached, so a re-run adds ≈ only answer-generation. Per-config marginal "
        "cost isn't broken out — the LLM token counter is cumulative across the sweep._",
        "",
        "## Caveats",
        "",
        "- **Retrieval P/R/MRR carries pooling bias** — the test set was labeled from the baseline "
        "config's pool, so non-baseline configs can't fully win on retrieval metrics. The ranking "
        "leads on faithfulness / answer-relevance, which judge the answer against its own context "
        "and are immune to this. Union-pooling is deferred to Ship I.",
        "- **OFAT, not a full grid** — one knob varied at a time around the baseline; interactions "
        "(e.g. chunk 512 only shining at top_k 10) aren't measured.",
        "- **The judge is an LLM** — faithfulness / answer-relevance are estimates with their own "
        "noise; the ranking treats gaps within ~0.01 (2 dp) as ties.",
        "",
    ]
    return "\n".join(lines)


def _run_sweep(args, settings, testset, out_dir) -> None:
    # Sweep builds its OWN throwaway indexes per config, so it does not go through
    # _build_components (which binds to canonical data/chroma/ and aborts if empty).
    embedder = EmbeddingGenerator(settings)
    llm = LLMClient(settings)
    db = Database()
    cache = JudgeCache(os.path.join(out_dir, "judge_cache.json"))

    configs = build_grid()
    n_builds = len({c.index_slug for c in configs})
    logger.info(
        "Sweeping %d configs over %d index build(s) on %d queries…",
        len(configs), n_builds, len(testset),
    )
    rows = run_sweep(testset, settings, db, embedder, llm, cache, configs=configs)
    ranked = rank_rows(rows)

    prompt_tokens = getattr(llm, "total_prompt_tokens", 0)
    completion_tokens = getattr(llm, "total_completion_tokens", 0)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "testset_path": args.testset,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "n_configs": len(configs),
        "n_index_builds": n_builds,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_cost_usd": _est_cost(prompt_tokens, completion_tokens),
    }
    stamp = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(out_dir, f"sweep_eval_{stamp}.md")
    json_path = os.path.join(out_dir, f"sweep_eval_{stamp}.json")
    markdown = build_sweep_markdown(ranked, meta)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "rows": [_sweep_row_scalars(r) for r in ranked]},
            f, ensure_ascii=False, indent=2,
        )
    print("\n" + markdown)
    print(f"\nWrote {md_path} and {json_path}")


def main() -> None:
    # Windows console defaults to cp1252, which can't encode report chars like ≈;
    # force UTF-8 so the printed Markdown matches what we write to disk.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ship F/G — score retrieval and generation against the test set.")
    parser.add_argument("--top-k", type=int, default=None, help="served depth (default RETRIEVAL_TOP_K)")
    parser.add_argument("--k-sweep", type=str, default="1,3,5,10", help="comma-separated k values (retrieval)")
    parser.add_argument("--judge", action="store_true", help="run generation eval (faithfulness + answer-relevance) instead of retrieval")
    parser.add_argument("--sweep", action="store_true", help="run the Ship H multi-config OFAT sweep (builds throwaway per-config indexes)")
    parser.add_argument("--gen-sample", type=int, default=None, help="cap in-domain queries judged (generation only)")
    parser.add_argument("--testset", type=str, default=TESTSET_PATH)
    parser.add_argument("--out", type=str, default=None, help="output dir (default <OUTPUT_DIR>/eval)")
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    out_dir = args.out or os.path.join(settings.OUTPUT_DIR, "eval")

    testset = load_testset(args.testset)
    if not testset:
        logger.error("Test set %s is empty — nothing to evaluate.", args.testset)
        return

    os.makedirs(out_dir, exist_ok=True)
    if args.sweep:
        _run_sweep(args, settings, testset, out_dir)
        return

    components = _build_components(settings)
    if components is None:
        return

    if args.judge:
        _run_generation(args, settings, testset, out_dir, components)
    else:
        _run_retrieval(args, settings, testset, out_dir, components)


if __name__ == "__main__":
    main()
