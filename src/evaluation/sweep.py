"""src/evaluation/sweep.py — Ship H: OFAT multi-config comparison runner.

Sweep a small one-factor-at-a-time grid of retrieval/index configs around the
default baseline (chunk 256, top_k 5, MiniLM), run BOTH eval halves per config
(Ship F retrieval P/R/MRR + latency, Ship G faithfulness + answer-relevance +
cost), and collect one comparison row per config so evaluate.py --sweep can rank
them and doc/ship-h-findings.md can name a winner.

Forks resolved 2026-06-26:
  - FORK A: embedding model held FIXED at all-MiniLM-L6-v2 (mpnet cut). So the only
    expensive axis is chunk_size → 3 index builds, not 6.
  - FORK B: union-pooling DEFERRED to Ship I. Retrieval P/R is reported with the
    standing pooling-bias caveat; the headline config call leans on the bias-free
    generation metrics (faithfulness/relevance).

Cost structure that drives the design:
  - chunk_size change  → EXPENSIVE: re-chunk + re-embed the whole corpus into a
    fresh ChromaDB persist dir.
  - top_k change       → FREE: query-time only; the stored index is unchanged.
  So: build one index per UNIQUE (chunk_size, embedding_model); loop top_k innermost
  over that already-built index. evaluate_retrieval() already k-sweeps in a single
  call, so it runs ONCE per index; evaluate_generation() takes a single top_k, so it
  runs ONCE per top_k.

Read-only over the canonical corpus: reads articles from SQLite read-only, writes
ONLY to data/chroma_sweep/<index-slug>/ (gitignored under data/). NEVER touches
data/chroma/ or data/news.db. The judge cache + reports are the only other writes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import Settings
from src.evaluation.harness import (
    EvalReport,
    GenerationReport,
    evaluate_generation,
    evaluate_retrieval,
)
from src.evaluation.testset import TestQuery
from src.rag.chunker import chunk_article
from src.rag.qa import QAEngine
from src.rag.retriever import Retriever
from src.storage.vector_store import VectorStore

# gpt-4o-mini list price (USD per 1M tokens) — mirrors evaluate.py._est_cost so the
# sweep stays self-contained (importing the CLI module would be circular).
GPT_4O_MINI_INPUT_PER_1M = 0.15
GPT_4O_MINI_OUTPUT_PER_1M = 0.60


def _est_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1e6 * GPT_4O_MINI_INPUT_PER_1M
        + completion_tokens / 1e6 * GPT_4O_MINI_OUTPUT_PER_1M
    )


def _model_slug(embedding_model: str) -> str:
    """Filesystem-safe short name: drop any 'org/' prefix, e.g.
    'sentence-transformers/all-MiniLM-L6-v2' -> 'all-MiniLM-L6-v2'."""
    return embedding_model.rsplit("/", 1)[-1]

# Baseline = the served default config (bold values in the plan's OFAT grid).
BASELINE_CHUNK_SIZE = 256
BASELINE_OVERLAP = 38
BASELINE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASELINE_TOP_K = 5

# OFAT axis values. Baseline value appears in each so build_grid can dedup it.
CHUNK_SIZES = (128, 256, 512)
TOP_KS = (3, 5, 10)

# Where throwaway per-config indexes live (under the already-gitignored data/).
SWEEP_ROOT = os.path.join("data", "chroma_sweep")


@dataclass(frozen=True)
class SweepConfig:
    """One point in the OFAT grid. Two configs that differ ONLY in top_k share the
    same physical index (see index_slug)."""
    chunk_size: int
    overlap: int
    embedding_model: str
    top_k: int

    @property
    def index_slug(self) -> str:
        """Identity of the *stored index* — (chunk_size, embedding_model) only.
        top_k is deliberately excluded: it's a free query-time knob, so configs that
        differ only in top_k must map to the SAME persist dir (build once, reuse).

        TODO: return a filesystem-safe slug, e.g. f"chunk{chunk_size}_{model_slug}".
        """
        return f"chunk{self.chunk_size}_{_model_slug(self.embedding_model)}"

    @property
    def label(self) -> str:
        """Human-readable row label for the comparison table, e.g.
        "chunk256 / top_k5 / MiniLM". Includes top_k (unlike index_slug)."""
        model = "MiniLM" if "minilm" in self.embedding_model.lower() else _model_slug(self.embedding_model)
        return f"chunk{self.chunk_size} / top_k{self.top_k} / {model}"


def build_grid(
    chunk_sizes=CHUNK_SIZES,
    top_ks=TOP_KS,
    embedding_model: str = BASELINE_EMBEDDING_MODEL,
    overlap: int = BASELINE_OVERLAP,
) -> List[SweepConfig]:
    """Enumerate the OFAT grid around the baseline. Vary ONE axis at a time:

      - chunk_size ∈ chunk_sizes, top_k = BASELINE_TOP_K   (3 configs; 3 index builds)
      - top_k ∈ top_ks, chunk_size = BASELINE_CHUNK_SIZE   (3 configs; 0 new builds)

    The baseline (chunk 256, top_k 5) is shared by both arms — emit it exactly ONCE.
    embedding_model is fixed (Fork A). Net: 5 distinct configs over 3 index builds.

    NOTE: overlap should scale with chunk_size to keep ~15% (e.g. round(0.15*size)),
    rather than pinning the baseline's 38 onto the 128/512 indexes — decide and
    document which, since it affects chunk boundaries.

    TODO: build the deduped list of SweepConfig. Order it so build_index runs once
    per index_slug (group by chunk_size).
    """
    # Overlap scales proportionally with chunk_size off the baseline (38 tok @ 256 ≈
    # 15%), so the 128/512 indexes keep the same ~15% boundary feel rather than
    # inheriting the baseline's absolute 38.
    def _overlap(chunk_size: int) -> int:
        return round(overlap * chunk_size / BASELINE_CHUNK_SIZE)

    # OFAT: vary chunk_size at the baseline top_k, then top_k at the baseline chunk.
    pairs = [(cs, BASELINE_TOP_K) for cs in chunk_sizes]
    pairs += [(BASELINE_CHUNK_SIZE, tk) for tk in top_ks]

    seen = set()
    configs: List[SweepConfig] = []
    for cs, tk in pairs:
        cfg = SweepConfig(
            chunk_size=cs,
            overlap=_overlap(cs),
            embedding_model=embedding_model,
            top_k=tk,
        )
        if cfg not in seen:  # the baseline (256, top_k5) lands in both arms — emit once
            seen.add(cfg)
            configs.append(cfg)

    # Group by chunk_size so build_index runs once per index_slug; top_k is secondary.
    configs.sort(key=lambda c: (c.chunk_size, c.top_k))
    return configs


def build_index(
    config: SweepConfig,
    settings: Settings,
    db,
    embedder,
    sweep_root: str = SWEEP_ROOT,
):
    """Build (or reuse) the throwaway chunk-level index for `config` and return a
    Retriever bound to it.

    Mirrors the pipeline.py indexing block (lines ~101–122) but: reads the WHOLE
    corpus (db.get_all_articles(), read-only), chunks at config.chunk_size /
    config.overlap, embeds with the shared MiniLM `embedder`, and writes to an
    ISOLATED persist dir data/chroma_sweep/<config.index_slug>/.

    Index isolation without forking VectorStore: clone settings with the per-config
    dir, then point a VectorStore at it —
        cfg_settings = settings.model_copy(update={"CHROMA_PERSIST_DIR": dir})
        vstore = VectorStore(cfg_settings)
    Different path ⇒ different ChromaDB ⇒ canonical data/chroma/ is untouched.

    Resumable: if the persist dir already exists AND its collection count > 0, SKIP
    the rebuild and just bind a Retriever to it.

    Steps when building:
      1. dir = os.path.join(sweep_root, config.index_slug)
      2. rows = db.get_all_articles()  # ORM rows, read-only
      3. for each row: convert to the chunk_article dict (id/content/title/source/
         url/published_at — see pipeline.py:104), chunk with embedder.model.tokenizer
         at config.chunk_size / config.overlap, collect chunks.
      4. embeddings = embedder.generate_embeddings([c["text"] for c in chunks]).tolist()
      5. vstore.add_chunks(chunks, embeddings)
      6. return Retriever(embedder, vstore)

    Embedder reuse: MiniLM is fixed (Fork A), so the SAME embedder serves every
    config. If the embedding-model axis is ever reopened, construct an
    EmbeddingGenerator per config.embedding_model here instead.

    TODO: implement; never write outside sweep_root.
    """
    persist_dir = os.path.join(sweep_root, config.index_slug)
    os.makedirs(persist_dir, exist_ok=True)

    # Isolate ChromaDB to the throwaway dir — canonical data/chroma/ is never opened.
    cfg_settings = settings.model_copy(update={"CHROMA_PERSIST_DIR": persist_dir})
    vstore = VectorStore(cfg_settings)

    # Resumable: a populated collection means this (chunk_size, model) index already
    # exists from a prior run — reuse it instead of re-chunking + re-embedding.
    if vstore.financial_news_collection.count() > 0:
        return Retriever(embedder, vstore)

    chunks: List[dict] = []
    for row in db.get_all_articles():  # read-only ORM rows over the whole corpus
        article = {
            "id": row.id,
            "content": row.content,
            "title": row.title,
            "source": row.source,
            "url": row.url,
            "published_at": row.published_at,
        }
        chunks.extend(
            chunk_article(
                article,
                embedder.model.tokenizer,
                config.chunk_size,
                config.overlap,
            )
        )

    if chunks:
        embeddings = embedder.generate_embeddings([c["text"] for c in chunks]).tolist()
        vstore.add_chunks(chunks, embeddings)

    return Retriever(embedder, vstore)


@dataclass
class SweepRow:
    """One config's full result line for the comparison table."""
    config: SweepConfig
    retrieval: EvalReport          # k-swept; read metrics at config.top_k
    generation: GenerationReport   # judged at config.top_k
    # Convenience scalars pulled out for ranking/printing (fill from the reports):
    precision: Optional[float] = None
    recall: Optional[float] = None
    hit_rate: Optional[float] = None
    mrr: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    est_cost_usd: float = 0.0


def run_sweep(
    testset: List[TestQuery],
    settings: Settings,
    db,
    embedder,
    llm,
    cache,
    configs: Optional[List[SweepConfig]] = None,
) -> List[SweepRow]:
    """Run the whole sweep and return one SweepRow per config.

    Order to minimize re-indexing (the whole point):
      group configs by index_slug → for each unique index:
          retriever = build_index(any-config-with-this-slug, ...)   # ONCE
          qa_engine = QAEngine(retriever, llm)
          retr_report = evaluate_retrieval(testset, retriever, qa_engine,
                                           top_k=BASELINE_TOP_K, k_values=sorted top_ks)
              # k-swept in a single call → covers every top_k for this index
          for top_k in this index's configs (innermost, FREE):
              gen_report = evaluate_generation(testset, retriever, qa_engine, llm,
                                               embedder, top_k, cache)
              row = SweepRow(config, retr_report, gen_report, ...scalars at top_k...)

    Notes:
      - One shared `cache` (JudgeCache) across all configs is CORRECT: its key hashes
        (answer+context), which differ per chunk_size and per top_k, so each config
        re-judges — i.e. each pays a full generation-eval. Budget ~$0.05 × #configs.
      - Pull retrieval scalars from retr_report.per_k_means[config.top_k]; faithfulness
        / relevance from gen_report; est_cost via evaluate.py's _est_cost on the
        gen_report token totals (or recompute here).
      - Read-only over the corpus; only build_index writes (to sweep dirs).

    TODO: implement the grouped loop; return the rows.
    """
    if configs is None:
        configs = build_grid()

    # Group by index identity, preserving build_grid's order. Each group shares one
    # physical index; its configs differ only in the free top_k knob.
    groups: Dict[str, List[SweepConfig]] = {}
    for c in configs:
        groups.setdefault(c.index_slug, []).append(c)

    rows: List[SweepRow] = []
    for cfgs in groups.values():
        retriever = build_index(cfgs[0], settings, db, embedder)  # ONCE per slug
        qa_engine = QAEngine(retriever, llm)

        # k-sweep every top_k this index needs in a single retrieval pass.
        k_values = sorted({c.top_k for c in cfgs})
        retr_report = evaluate_retrieval(
            testset, retriever, qa_engine, top_k=BASELINE_TOP_K, k_values=k_values
        )

        for config in cfgs:  # innermost, FREE: query-time top_k only
            gen_report = evaluate_generation(
                testset, retriever, qa_engine, llm, embedder, config.top_k, cache
            )
            rows.append(_build_row(config, retr_report, gen_report))

    return rows


def _build_row(
    config: SweepConfig, retr_report: EvalReport, gen_report: GenerationReport
) -> SweepRow:
    """Pull the scalar fields for `config.top_k` out of the two reports."""
    pk = retr_report.per_k_means.get(config.top_k, {})
    return SweepRow(
        config=config,
        retrieval=retr_report,
        generation=gen_report,
        precision=pk.get("precision"),
        recall=pk.get("recall"),
        hit_rate=pk.get("hit"),
        mrr=retr_report.mean_mrr,
        faithfulness=gen_report.mean_faithfulness,
        answer_relevance=gen_report.mean_answer_relevance,
        latency_p50_ms=retr_report.latency_p50_ms,
        est_cost_usd=_est_cost(gen_report.prompt_tokens, gen_report.completion_tokens),
    )


def rank_rows(rows: List[SweepRow]) -> List[SweepRow]:
    """Return rows ordered best→worst for the findings table. Pure function over the
    scalar fields — no I/O, no index — so it unit-tests against hand-built SweepRows.

    Ranking key (Fork B): lead with the bias-free generation metrics, since retrieval
    recall carries pooling bias. Suggested primary sort = faithfulness, then
    answer-relevance, then hit_rate/MRR as tie-breakers; treat tiny gaps as ties
    (judge variance). Document the exact key in the findings doc.

    TODO: implement the sort; decide + document the tie threshold.
    """
    # Tie threshold = round to 2 decimals (~0.01 band) on the generation metrics, so
    # differences inside judge variance don't decide the ranking — they fall through
    # to the (exact) retrieval tie-breakers. None ranks worst.
    def _bucket(x: Optional[float]) -> float:
        return round(x, 2) if x is not None else -1.0

    def _raw(x: Optional[float]) -> float:
        return x if x is not None else -1.0

    return sorted(
        rows,
        key=lambda r: (
            -_bucket(r.faithfulness),
            -_bucket(r.answer_relevance),
            -_raw(r.hit_rate),
            -_raw(r.mrr),
        ),
    )
