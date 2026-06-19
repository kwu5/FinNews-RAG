"""src/evaluation/judge.py — Ship G: LLM-judge metrics (SKELETON — implement me).

Custom implementation of the RAGAS metric DEFINITIONS (no RAGAS/LangChain):
faithfulness and answer-relevance. Pure orchestration over an injected `llm`
(LLMClient) and `embedder` (EmbeddingGenerator) — NO database/ChromaDB import, so
these functions unit-test against mocks (see tests/test_judge.py).

`llm` must provide:  decompose_claims(answer) -> list[str]
                     verify_claims(context, claims) -> list[bool]
                     generate_candidate_questions(answer, n=3) -> (list[str], bool)
`embedder` must provide:  generate_embedding(text) -> np.ndarray  (L2-normalized)

Fill in each NotImplementedError. tests/test_judge.py is the spec — implement to green.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def faithfulness(answer: str, context: str, llm) -> Optional[float]:
    """Fraction of the answer's claims that are supported by `context`.
    """
    claims = llm.decompose_claims(answer)
    if not claims:    
        return None
    verdicts = llm.verify_claims(context, claims)
    return sum(verdicts) / len(claims)
    


def answer_relevance(query: str, answer: str, llm, embedder) -> float:
    """Mean cosine similarity between the original `query` and questions an LLM
    reverse-generates from the `answer`. High = the answer is on-topic/complete.    
    """
    questions, noncommittal = llm.generate_candidate_questions(answer)
    if noncommittal or not questions:
      return 0.0
    q_emb = embedder.generate_embedding(query)
    total = 0
    for q in questions:
      sim_i = _cosine(embedder.generate_embedding(q), q_emb)
      total += sim_i
    return total/len(questions)
