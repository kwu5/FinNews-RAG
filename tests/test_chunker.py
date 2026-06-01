"""Tests for the Ship C chunker — token-windowed passage splitting.

These use a deterministic word-level FakeTokenizer so the windowing math
(overlap, sequential indices, short -> 1 chunk) is exact and fast. Real-tokenizer
behavior is exercised by tests/smoke_ship_c.py and src/rag/chunker.py __main__.
"""

import pytest

from src.rag.chunker import chunk_article


class FakeTokenizer:
    """One token per whitespace word; decode rejoins with spaces.

    Mirrors the slice of the HuggingFace tokenizer API that chunk_article uses:
        __call__(text, add_special_tokens=...) -> {"input_ids": [int, ...]}
        decode(ids, skip_special_tokens=...)   -> str
    """

    def __init__(self):
        self._to_id = {}
        self._to_word = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self._to_id:
                idx = len(self._to_id) + 1
                self._to_id[w] = idx
                self._to_word[idx] = w
            ids.append(self._to_id[w])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._to_word[i] for i in ids)


@pytest.fixture
def tok():
    return FakeTokenizer()


def _article(content, **over):
    base = {
        "id": 7,
        "content": content,
        "title": "T",
        "source": "S",
        "url": "https://e.com/a",
        "published_at": "2026-06-01T00:00:00Z",
    }
    base.update(over)
    return base


class TestChunkArticle:

    def test_short_article_one_chunk(self, tok):
        chunks = chunk_article(_article("one two three four five"), tok, 256, 38)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["text"] == "one two three four five"

    def test_exact_fit_one_chunk(self, tok):
        # exactly chunk_size tokens -> still one chunk (loop breaks at the tail)
        art = _article(" ".join(f"w{i}" for i in range(4)))
        assert len(chunk_article(art, tok, chunk_size=4, overlap=1)) == 1

    def test_long_article_splits_with_overlap(self, tok):
        # 10 tokens, size 4, overlap 1 -> step 3 -> windows [0:4],[3:7],[6:10]
        art = _article(" ".join(f"w{i}" for i in range(10)))
        chunks = chunk_article(art, tok, chunk_size=4, overlap=1)

        assert len(chunks) == 3
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
        assert all(len(c["text"].split()) == 4 for c in chunks)
        # overlap: last word of chunk i == first word of chunk i+1
        for a, b in zip(chunks, chunks[1:]):
            assert a["text"].split()[-1] == b["text"].split()[0]

    def test_no_trailing_overlap_only_chunk(self, tok):
        # 7 tokens, size 4, overlap 1, step 3 -> [0:4],[3:7]; 2nd window hits tail
        art = _article(" ".join(f"w{i}" for i in range(7)))
        # must be 2, NOT 3 — no leftover overlap-only chunk
        assert len(chunk_article(art, tok, chunk_size=4, overlap=1)) == 2

    def test_empty_content_returns_empty(self, tok):
        assert chunk_article(_article(""), tok, 256, 38) == []

    def test_whitespace_only_returns_empty(self, tok):
        assert chunk_article(_article("   \n\t  "), tok, 256, 38) == []

    def test_missing_content_key_returns_empty(self, tok):
        art = _article("x")
        del art["content"]
        assert chunk_article(art, tok, 256, 38) == []

    def test_overlap_ge_chunk_size_raises(self, tok):
        with pytest.raises(ValueError):
            chunk_article(_article("a b c"), tok, chunk_size=4, overlap=4)

    def test_metadata_carried_through(self, tok):
        art = _article(
            "one two three four five six seven eight",
            id=42, title="Fed", source="Reuters",
            url="https://e.com/x", published_at="2026-06-01T10:00:00Z",
        )
        chunks = chunk_article(art, tok, chunk_size=3, overlap=1)
        assert len(chunks) > 1
        for c in chunks:
            assert c["article_id"] == 42
            assert c["title"] == "Fed"
            assert c["source"] == "Reuters"
            assert c["url"] == "https://e.com/x"
            assert c["published_at"] == "2026-06-01T10:00:00Z"
            assert set(c) == {
                "chunk_index", "text", "article_id",
                "title", "source", "url", "published_at",
            }
