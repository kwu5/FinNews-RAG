"""src/evaluation/judge_cache.py — Ship G: judge-output cache (SKELETON — implement me).

A tiny JSON-on-disk float cache so re-running the LLM-judge is free + deterministic.
The key folds in hash(answer + context) so a CHANGED generation busts the entry
(stale scores never silently survive a re-index).

Caching contract (important):
  - Store only NON-None float scores. faithfulness() can return None (no claims);
    leave those UNCACHED so `get()` returning None unambiguously means "miss".
    No-claims rows are rare and cheap to recompute.

Fill in each NotImplementedError. tests/test_judge.py exercises this class.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional


class JudgeCache:
    def __init__(self, path: str) -> None:
        """Load the JSON dict at `path` into memory (missing/empty file -> {}).
        Remember `path` for save()."""
        self.path = path
        self._cache: Dict[str, float] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._cache = data
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def key(self, query_id: str, metric: str, answer: str, context: str = "") -> str:
        """Build a stable cache key. Combine query_id + metric with a sha256
        hexdigest of f"{answer}\\x00{context}" so a changed answer or context
        produces a different key."""
        digest = hashlib.sha256(f"{answer}\x00{context}".encode("utf-8")).hexdigest()
        return f"{query_id}:{metric}:{digest}"

    def get(self, key: str) -> Optional[float]:
        """Return the cached float for `key`, or None if absent."""
        return self._cache.get(key)

    def set(self, key: str, value: float) -> None:
        """Store `value` (a float) under `key` in memory."""
        self._cache[key] = float(value)

    def save(self) -> None:
        """Flush the in-memory dict to `path` as JSON (create parent dir if needed)."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
