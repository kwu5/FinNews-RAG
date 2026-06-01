"""Chunk articles into ~paragraph-sized passages for chunk-level embedding.

Why token-based (not word-split): all-MiniLM-L6-v2 truncates at 256 tokens, so
chunk sizing has to be measured in the model's *own* tokens to line up with what
the model actually sees. We use the embedding model's tokenizer (BERT WordPiece)
to slice on token boundaries, then decode each window back to text for embedding.
"""

from typing import Any, List


def chunk_article(
    article: dict,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> List[dict]:
    """Split one article's content into overlapping token-windowed chunks.

    Args:
        article: a saved article. Reads "id" (the SQLite autoincrement, used as
            article_id), "content" (the text to chunk), and metadata keys
            "title", "source", "url", "published_at".
        tokenizer: the embedding model's tokenizer, e.g.
            EmbeddingGenerator.model.tokenizer — reused so the model isn't loaded
            twice and token counts match the embedder exactly.
        chunk_size: max content tokens per chunk (e.g. CHUNK_SIZE_TOKENS).
        overlap: tokens carried from the tail of one chunk into the next
            (e.g. CHUNK_OVERLAP_TOKENS).

    Returns:
        A list of chunk dicts, each:
            {chunk_index, text, article_id, title, source, url, published_at}
        - short article (<= chunk_size tokens) -> exactly one chunk
        - empty / whitespace-only content -> [] (nothing to index)
        - chunk_index is sequential from 0
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
            "otherwise the window never advances."
        )

    content = (article.get("content") or "").strip()
    if not content:
        return []

    # add_special_tokens=False: count only real content tokens. The model adds
    # its own [CLS]/[SEP] at encode time, so we slice on content tokens here.
    token_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    if not token_ids:
        return []

    step = chunk_size - overlap
    n = len(token_ids)

    chunks: List[dict] = []
    start = 0
    chunk_index = 0
    while start < n:
        window = token_ids[start : start + chunk_size]
        text = tokenizer.decode(window, skip_special_tokens=True).strip()
        if text:  # decode can yield empty on a window of only stripped tokens
            chunks.append(_make_chunk(article, chunk_index, text))
            chunk_index += 1
        if start + chunk_size >= n:  # this window reached the tail; stop
            break
        start += step

    return chunks


def _make_chunk(article: dict, chunk_index: int, text: str) -> dict:
    """Build one chunk dict, carrying article_id + metadata through."""
    return {
        "chunk_index": chunk_index,
        "text": text,
        "article_id": article["id"],
        "title": article.get("title"),
        "source": article.get("source"),
        "url": article.get("url"),
        "published_at": article.get("published_at"),
    }


if __name__ == "__main__":
    # Quick eyeball test against the real tokenizer.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    tok = model.tokenizer

    short = {
        "id": 1,
        "title": "Fed holds rates",
        "source": "Reuters",
        "url": "https://example.com/a",
        "published_at": "2026-06-01T10:00:00Z",
        "content": "The Federal Reserve left rates unchanged on Wednesday.",
    }
    long = {
        "id": 2,
        "title": "Long market wrap",
        "source": "CNBC",
        "url": "https://example.com/b",
        "published_at": "2026-06-01T11:00:00Z",
        "content": "Markets rallied. " * 200,  # ~hundreds of tokens
    }

    short_chunks = chunk_article(short, tok, chunk_size=256, overlap=38)
    long_chunks = chunk_article(long, tok, chunk_size=256, overlap=38)

    print(f"short -> {len(short_chunks)} chunk(s)")
    assert len(short_chunks) == 1, "short article should be exactly 1 chunk"

    print(f"long  -> {len(long_chunks)} chunk(s)")
    assert len(long_chunks) > 1, "long article should split into multiple chunks"
    assert [c["chunk_index"] for c in long_chunks] == list(range(len(long_chunks)))
    assert all(c["article_id"] == 2 for c in long_chunks)
    print("OK")
