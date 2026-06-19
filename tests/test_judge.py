"""Spec for the Ship G LLM-judge (src/evaluation/judge.py + judge_cache.py).

These FAIL until the two skeleton files are implemented — they are the executable
definition of done. Pure logic only: the LLM and embedder are mocked, so no API
calls and no ChromaDB.

  - judge._cosine        — similarity on known vectors
  - judge.faithfulness   — supported/total; None when there are no claims
  - judge.answer_relevance — mean cosine of reverse-questions; 0 when noncommittal
  - JudgeCache           — set/get, key busting, persistence
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation import judge
from src.evaluation.judge_cache import JudgeCache


class TestCosine:
    def test_identical_is_one(self):
        a = np.array([1.0, 0.0])
        assert judge._cosine(a, a) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert judge._cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_unnormalized_inputs_normalized(self):
        # same direction, different magnitude -> still 1.0
        assert judge._cosine(np.array([2.0, 0.0]), np.array([5.0, 0.0])) == pytest.approx(1.0)

    def test_zero_vector_guarded(self):
        assert judge._cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def _faith_llm(claims, verdicts):
    llm = MagicMock()
    llm.decompose_claims.return_value = claims
    llm.verify_claims.return_value = verdicts
    return llm


class TestFaithfulness:
    def test_partial_support(self):
        llm = _faith_llm(["c1", "c2", "c3"], [True, True, False])
        assert judge.faithfulness("ans", "ctx", llm) == pytest.approx(2 / 3)

    def test_no_claims_returns_none_and_skips_verify(self):
        llm = _faith_llm([], [])
        assert judge.faithfulness("ans", "ctx", llm) is None
        llm.verify_claims.assert_not_called()  # short-circuit before verifying

    def test_all_unsupported_is_zero(self):
        llm = _faith_llm(["c1", "c2"], [False, False])
        assert judge.faithfulness("ans", "ctx", llm) == 0.0

    def test_all_supported_is_one(self):
        llm = _faith_llm(["c1", "c2"], [True, True])
        assert judge.faithfulness("ans", "ctx", llm) == 1.0


def _embedder(mapping):
    """Mock embedder: text -> fixed vector."""
    emb = MagicMock()
    emb.generate_embedding.side_effect = lambda t: np.array(mapping[t], dtype=float)
    return emb


class TestAnswerRelevance:
    def test_mean_cosine_of_reverse_questions(self):
        llm = MagicMock()
        llm.generate_candidate_questions.return_value = (["q1", "q2"], False)
        emb = _embedder({
            "the query": [1.0, 0.0],
            "q1": [1.0, 0.0],   # cosine 1.0 to query
            "q2": [0.0, 1.0],   # cosine 0.0 to query
        })
        assert judge.answer_relevance("the query", "ans", llm, emb) == pytest.approx(0.5)

    def test_noncommittal_is_zero_without_embedding(self):
        llm = MagicMock()
        llm.generate_candidate_questions.return_value = (["q1"], True)
        emb = MagicMock()
        assert judge.answer_relevance("q", "ans", llm, emb) == 0.0
        emb.generate_embedding.assert_not_called()

    def test_no_questions_is_zero(self):
        llm = MagicMock()
        llm.generate_candidate_questions.return_value = ([], False)
        emb = MagicMock()
        assert judge.answer_relevance("q", "ans", llm, emb) == 0.0


class TestJudgeCache:
    def test_set_get_roundtrip(self, tmp_path):
        c = JudgeCache(str(tmp_path / "jc.json"))
        k = c.key("q1", "faithfulness", "answer text", "context")
        assert c.get(k) is None       # miss before set
        c.set(k, 0.75)
        assert c.get(k) == 0.75

    def test_key_stable_for_same_inputs(self, tmp_path):
        c = JudgeCache(str(tmp_path / "jc.json"))
        assert c.key("q1", "faithfulness", "a", "ctx") == c.key("q1", "faithfulness", "a", "ctx")

    def test_key_busts_on_changed_answer(self, tmp_path):
        c = JudgeCache(str(tmp_path / "jc.json"))
        assert c.key("q1", "faithfulness", "answer A", "ctx") != c.key("q1", "faithfulness", "answer B", "ctx")

    def test_key_busts_on_changed_context(self, tmp_path):
        c = JudgeCache(str(tmp_path / "jc.json"))
        assert c.key("q1", "faithfulness", "a", "ctx1") != c.key("q1", "faithfulness", "a", "ctx2")

    def test_persists_across_save_and_reload(self, tmp_path):
        path = str(tmp_path / "jc.json")
        c = JudgeCache(path)
        k = c.key("q2", "answer_relevance", "ans", "")
        c.set(k, 0.4)
        c.save()
        assert JudgeCache(path).get(k) == 0.4

    def test_missing_file_starts_empty(self, tmp_path):
        c = JudgeCache(str(tmp_path / "nope.json"))
        assert c.get("anything") is None
