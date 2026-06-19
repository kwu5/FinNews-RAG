"""Tests for the Ship F retrieval-eval harness.

Three layers, all without touching ChromaDB or OpenAI:
  - metrics.py: pure functions on hand-built (ranked ids, relevant set) fixtures;
  - testset.py: loader round-trips a temp JSONL into typed rows;
  - harness.py: a mocked retriever/qa_engine drives evaluate_retrieval, covering
    chunk->article dedup, the in/out-of-domain split, and abstention scoring.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.evaluation import judge, metrics
from src.evaluation.harness import dedup_to_articles, evaluate_generation, evaluate_retrieval
from src.evaluation.testset import TestQuery, load_testset


class TestMetrics:
    def test_perfect_single_relevant(self):
        assert metrics.precision([1], {1}) == 1.0
        assert metrics.recall([1], {1}) == 1.0
        assert metrics.hit_rate([1], {1}) == 1.0
        assert metrics.reciprocal_rank([1], {1}) == 1.0

    def test_complete_miss(self):
        assert metrics.precision([2, 3], {1}) == 0.0
        assert metrics.recall([2, 3], {1}) == 0.0
        assert metrics.hit_rate([2, 3], {1}) == 0.0
        assert metrics.reciprocal_rank([2, 3], {1}) == 0.0

    def test_partial_ranked(self):
        ranked, relevant = [2, 1, 3], {1, 4}
        assert metrics.precision(ranked, relevant) == 1 / 3
        assert metrics.recall(ranked, relevant) == 1 / 2
        assert metrics.hit_rate(ranked, relevant) == 1.0
        assert metrics.reciprocal_rank(ranked, relevant) == 1 / 2  # relevant at rank 2

    def test_empty_retrieval_is_zero_not_error(self):
        assert metrics.precision([], {1}) == 0.0
        assert metrics.recall([], {1}) == 0.0
        assert metrics.hit_rate([], {1}) == 0.0
        assert metrics.reciprocal_rank([], {1}) == 0.0

    def test_empty_relevant_recall_guarded(self):
        assert metrics.recall([1, 2], set()) == 0.0

    def test_abstention_correct_truth_table(self):
        assert metrics.abstention_correct(False, is_out_of_domain=True) is True
        assert metrics.abstention_correct(True, is_out_of_domain=True) is False
        assert metrics.abstention_correct(True, is_out_of_domain=False) is True
        assert metrics.abstention_correct(False, is_out_of_domain=False) is False


class TestLoader:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "ts.jsonl"
        p.write_text(
            json.dumps({"query_id": "q001", "query": "a?", "relevant_article_ids": [3, 7],
                        "source": "llm", "type": "in_domain", "notes": ""}) + "\n"
            + "\n"  # blank line skipped
            + json.dumps({"query_id": "q002", "query": "b?", "relevant_article_ids": [],
                          "source": "hand", "type": "out_of_domain", "notes": "abstain"}) + "\n",
            encoding="utf-8",
        )
        rows = load_testset(str(p))
        assert len(rows) == 2
        assert rows[0] == TestQuery("q001", "a?", {3, 7}, "llm", "in_domain", "")
        assert rows[0].relevant_article_ids == {3, 7}
        assert rows[1].is_out_of_domain is True


def _chunk(article_id, distance=0.1):
    return {"article_id": article_id, "distance": distance}


class TestDedup:
    def test_keeps_first_per_article_in_order(self):
        hits = [_chunk(5), _chunk(5), _chunk(7), _chunk(5), _chunk(2)]
        assert dedup_to_articles(hits) == [5, 7, 2]


class TestHarness:
    def _retriever(self, hits):
        r = MagicMock()
        r.retrieve.return_value = hits
        return r

    def test_in_domain_metrics_and_dedup(self):
        # top chunks: article 9 (twice), then 4 — relevant is {4}
        hits = [_chunk(9), _chunk(9), _chunk(4), _chunk(8)]
        retriever = self._retriever(hits)
        qa = MagicMock()
        ts = [TestQuery("q1", "q", {4}, "llm", "in_domain", "")]

        report = evaluate_retrieval(ts, retriever, qa, top_k=3, k_values=[1, 3])

        assert report.n_in_domain == 1 and report.n_out_of_domain == 0
        r = report.results[0]
        # k=1 -> chunks[:1] = [9] -> articles [9]; relevant {4} -> all zero
        assert r.per_k[1] == {"precision": 0.0, "recall": 0.0, "hit": 0.0}
        # k=3 -> chunks[:3] = [9,9,4] -> articles [9,4]; 1 of 2 relevant
        assert r.per_k[3]["precision"] == 0.5
        assert r.per_k[3]["recall"] == 1.0
        assert r.per_k[3]["hit"] == 1.0
        # MRR at deepest k: articles [9,4,8] -> relevant 4 at rank 2
        assert r.mrr == 0.5
        assert r.latency_ms is not None
        qa.answer_query.assert_not_called()  # in-domain never calls the LLM

    def test_out_of_domain_uses_qa_and_scores_abstention(self):
        retriever = self._retriever([_chunk(1)])
        qa = MagicMock()
        qa.answer_query.return_value = SimpleNamespace(answered_from_context=False)
        ts = [TestQuery("q9", "off-topic", set(), "hand", "out_of_domain", "")]

        report = evaluate_retrieval(ts, retriever, qa, top_k=5, k_values=[5])

        assert report.n_out_of_domain == 1
        assert report.abstention_accuracy == 1.0  # abstained correctly
        assert report.results[0].abstention_correct is True
        qa.answer_query.assert_called_once()

    def test_ood_that_answers_is_scored_wrong(self):
        retriever = self._retriever([_chunk(1)])
        qa = MagicMock()
        qa.answer_query.return_value = SimpleNamespace(answered_from_context=True)
        ts = [TestQuery("q9", "off-topic", set(), "hand", "out_of_domain", "")]

        report = evaluate_retrieval(ts, retriever, qa, top_k=5, k_values=[5])
        assert report.abstention_accuracy == 0.0


class _FakeCache:
    """Duck-typed JudgeCache for harness tests (no disk, no real key hashing)."""
    def __init__(self):
        self.d = {}
        self.saved = False
    def key(self, query_id, metric, answer, context=""):
        return f"{query_id}|{metric}|{answer}|{context}"
    def get(self, key):
        return self.d.get(key)
    def set(self, key, value):
        self.d[key] = value
    def save(self):
        self.saved = True


class TestGenerationHarness:
    def _qa(self, answered_for):
        """qa whose answer_query reports answered_from_context based on the query."""
        qa = MagicMock()
        qa.answer_query.side_effect = lambda query, k: SimpleNamespace(
            answer=f"A::{query}", answered_from_context=(query in answered_for)
        )
        return qa

    def test_in_out_split_and_scoring(self, monkeypatch):
        # one answered in-domain, one abstaining in-domain, one OOD (must be ignored)
        ts = [
            TestQuery("q1", "answerable", {1}, "llm", "in_domain", ""),
            TestQuery("q2", "abstains", {2}, "llm", "in_domain", ""),
            TestQuery("q9", "ood", set(), "hand", "out_of_domain", ""),
        ]
        retriever = MagicMock()
        retriever.retrieve.return_value = [{"text": "ctx chunk"}]
        qa = self._qa(answered_for={"answerable"})
        monkeypatch.setattr(judge, "faithfulness", lambda answer, context, llm: 1.0)
        monkeypatch.setattr(judge, "answer_relevance", lambda query, answer, llm, embedder: 0.8)

        llm = SimpleNamespace(total_prompt_tokens=100, total_completion_tokens=20)
        cache = _FakeCache()
        report = evaluate_generation(ts, retriever, qa, llm, MagicMock(), top_k=5, cache=cache)

        assert report.n_in_domain == 2          # OOD excluded entirely
        assert report.n_answered == 1
        assert report.n_skipped_unanswered == 1
        assert report.mean_faithfulness == 1.0
        assert report.mean_answer_relevance == 0.8
        assert report.prompt_tokens == 100 and report.completion_tokens == 20
        assert cache.saved is True

    def test_sample_caps_in_domain(self, monkeypatch):
        ts = [TestQuery(f"q{i}", f"query{i}", {i}, "llm", "in_domain", "") for i in range(5)]
        retriever = MagicMock()
        retriever.retrieve.return_value = [{"text": "c"}]
        qa = self._qa(answered_for={f"query{i}" for i in range(5)})
        monkeypatch.setattr(judge, "faithfulness", lambda answer, context, llm: 1.0)
        monkeypatch.setattr(judge, "answer_relevance", lambda query, answer, llm, embedder: 1.0)

        llm = SimpleNamespace(total_prompt_tokens=0, total_completion_tokens=0)
        report = evaluate_generation(ts, retriever, qa, llm, MagicMock(), top_k=5, cache=_FakeCache(), sample=2)
        assert report.n_in_domain == 2

    def test_cache_hit_avoids_judge_call(self, monkeypatch):
        ts = [TestQuery("q1", "answerable", {1}, "llm", "in_domain", "")]
        retriever = MagicMock()
        retriever.retrieve.return_value = [{"text": "ctx"}]
        qa = self._qa(answered_for={"answerable"})

        calls = {"f": 0, "r": 0}
        monkeypatch.setattr(judge, "faithfulness", lambda a, c, l: calls.__setitem__("f", calls["f"] + 1) or 1.0)
        monkeypatch.setattr(judge, "answer_relevance", lambda q, a, l, e: calls.__setitem__("r", calls["r"] + 1) or 0.5)

        llm = SimpleNamespace(total_prompt_tokens=0, total_completion_tokens=0)
        cache = _FakeCache()
        # Pre-seed both metrics with the exact key the harness will compute.
        ctx = "[1] ctx\n\n"
        cache.set(cache.key("q1", "faithfulness", "A::answerable", ctx), 0.9)
        cache.set(cache.key("q1", "answer_relevance", "A::answerable", ""), 0.7)

        report = evaluate_generation(ts, retriever, qa, llm, MagicMock(), top_k=5, cache=cache)
        assert calls == {"f": 0, "r": 0}        # both served from cache, no judge calls
        assert report.mean_faithfulness == 0.9
        assert report.mean_answer_relevance == 0.7
