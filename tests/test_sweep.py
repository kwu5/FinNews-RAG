"""Spec for the Ship H sweep runner (src/evaluation/sweep.py).

These FAIL until sweep.py's stubs are implemented — they are the executable
definition of done. Pure logic only: no real index build, no API, no ChromaDB.
build_index / evaluate_retrieval / evaluate_generation are monkeypatched so the
runner's *orchestration* (grid shape, build-once-per-index, top_k innermost,
ranking) is what's under test, not the heavy lifting.

  - build_grid       — OFAT shape: baseline emitted once, one axis varied at a time
  - SweepConfig.index_slug — identity excludes top_k (configs sharing an index)
  - run_sweep        — builds one index per unique slug, varies top_k for free
  - rank_rows        — orders configs best→worst on the (bias-free) generation metrics
"""

from unittest.mock import MagicMock

import pytest

from src.evaluation import sweep
from src.evaluation.harness import EvalReport, GenerationReport
from src.evaluation.sweep import SweepConfig, SweepRow


def _cfg(chunk_size=256, top_k=5, overlap=38, model="all-MiniLM-L6-v2") -> SweepConfig:
    return SweepConfig(chunk_size=chunk_size, overlap=overlap, embedding_model=model, top_k=top_k)


def _stub_eval_report(k_values) -> EvalReport:
    """Cheap EvalReport with per_k_means populated for every swept k (run_sweep reads
    per_k_means[config.top_k])."""
    k_values = sorted(set(k_values))
    return EvalReport(
        top_k=5, k_values=list(k_values), n_in_domain=0, n_out_of_domain=0,
        per_k_means={k: {"precision": 0.0, "recall": 0.0, "hit": 0.0} for k in k_values},
        mean_mrr=0.0, abstention_accuracy=None, latency_p50_ms=1.0, latency_p95_ms=2.0,
    )


def _stub_gen_report(top_k) -> GenerationReport:
    return GenerationReport(
        top_k=top_k, n_in_domain=0, n_answered=0, n_skipped_unanswered=0,
        mean_faithfulness=0.9, n_faithfulness=0, mean_answer_relevance=0.8,
        n_answer_relevance=0, n_zero_relevance=0, prompt_tokens=0, completion_tokens=0,
    )


def _patch_harness(monkeypatch):
    """Stub both eval halves so run_sweep's orchestration is what's under test."""
    monkeypatch.setattr(
        sweep, "evaluate_retrieval",
        lambda testset, retriever, qa_engine, top_k, k_values: _stub_eval_report(k_values),
    )
    monkeypatch.setattr(
        sweep, "evaluate_generation",
        lambda testset, retriever, qa_engine, llm, embedder, top_k, cache, *a, **k: _stub_gen_report(top_k),
    )


class TestBuildGrid:
    def test_baseline_emitted_exactly_once(self):
        """The (chunk 256, top_k 5) baseline is shared by both OFAT arms — it must
        appear once, not twice."""
        grid = sweep.build_grid()
        baselines = [c for c in grid if c.chunk_size == 256 and c.top_k == 5]
        assert len(baselines) == 1

    def test_is_ofat_not_full_grid(self):
        """Every non-baseline config differs from baseline in exactly ONE axis
        (chunk_size XOR top_k) — never both."""
        grid = sweep.build_grid()
        for c in grid:
            varied = (c.chunk_size != 256) + (c.top_k != 5)
            assert varied <= 1, f"{c} varies more than one axis"

    def test_embedding_model_fixed(self):
        """Fork A: MiniLM is pinned across the whole grid."""
        grid = sweep.build_grid()
        assert {c.embedding_model for c in grid} == {"all-MiniLM-L6-v2"}

    def test_covers_all_axis_values(self):
        grid = sweep.build_grid()
        assert {c.chunk_size for c in grid} == {128, 256, 512}
        assert {c.top_k for c in grid} == {3, 5, 10}


class TestIndexSlug:
    def test_top_k_excluded_from_index_identity(self):
        """Two configs differing only in top_k share one physical index."""
        assert _cfg(top_k=3).index_slug == _cfg(top_k=10).index_slug

    def test_chunk_size_changes_index_identity(self):
        assert _cfg(chunk_size=128).index_slug != _cfg(chunk_size=512).index_slug


class TestRunSweep:
    def test_builds_one_index_per_unique_slug(self, monkeypatch):
        """The core efficiency claim: build_index runs once per (chunk_size, model),
        NOT once per config. With 3 chunk sizes that's 3 builds even though the grid
        has 5 configs (the baseline chunk is reused across the top_k sweep)."""
        build_calls = []

        def fake_build_index(config, *a, **k):
            build_calls.append(config.index_slug)
            return MagicMock(name="retriever")

        monkeypatch.setattr(sweep, "build_index", fake_build_index)
        _patch_harness(monkeypatch)

        rows = sweep.run_sweep(
            testset=[], settings=MagicMock(), db=MagicMock(),
            embedder=MagicMock(), llm=MagicMock(), cache=MagicMock(),
            configs=sweep.build_grid(),
        )

        assert len(set(build_calls)) == len(build_calls), "an index slug was built twice"
        assert len(set(build_calls)) == 3  # three chunk sizes → three builds

    def test_returns_one_row_per_config(self, monkeypatch):
        """One SweepRow per config (5), even though only 3 indexes were built."""
        monkeypatch.setattr(sweep, "build_index", lambda *a, **k: MagicMock())
        _patch_harness(monkeypatch)
        grid = sweep.build_grid()
        rows = sweep.run_sweep(
            testset=[], settings=MagicMock(), db=MagicMock(),
            embedder=MagicMock(), llm=MagicMock(), cache=MagicMock(),
            configs=grid,
        )
        assert len(rows) == len(grid)


class TestRankRows:
    def _row(self, faithfulness, answer_relevance, hit_rate=0.5) -> SweepRow:
        return SweepRow(
            config=_cfg(), retrieval=MagicMock(), generation=MagicMock(),
            faithfulness=faithfulness, answer_relevance=answer_relevance, hit_rate=hit_rate,
        )

    def test_orders_by_generation_metrics_first(self):
        """Fork B: lead on the bias-free generation metrics, not retrieval recall."""
        worse = self._row(faithfulness=0.80, answer_relevance=0.70)
        better = self._row(faithfulness=0.97, answer_relevance=0.82)
        ranked = sweep.rank_rows([worse, better])
        assert ranked[0] is better

    def test_tie_breaks_on_retrieval(self):
        """Equal generation metrics → fall back to hit_rate/MRR."""
        lo = self._row(faithfulness=0.95, answer_relevance=0.80, hit_rate=0.40)
        hi = self._row(faithfulness=0.95, answer_relevance=0.80, hit_rate=0.60)
        ranked = sweep.rank_rows([lo, hi])
        assert ranked[0] is hi
