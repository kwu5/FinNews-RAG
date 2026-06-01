import logging

from src.config import Settings
from src.ingestion.world_news_api import WorldNewsAPIClient
from src.ingestion.rss_reader import RSSReader
from src.processing.cleaner import TextCleaner
from src.processing.content_hash import compute_content_hash
from src.processing.embeddings import EmbeddingGenerator
from src.processing.deduplicator import Deduplicator
from src.processing.url_canon import canonicalize_url
from src.rag.chunker import chunk_article
from src.storage.database import Database
from src.storage.vector_store import VectorStore
from src.summarization.llm_client import LLMClient
from src.summarization.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


# ---------- Module-level component init (heavy models load once on import) ----------
settings = Settings()  # type: ignore

news_api = WorldNewsAPIClient(settings)
rss      = RSSReader()
cleaner  = TextCleaner()
embedder = EmbeddingGenerator(settings)
dedup    = Deduplicator(embedder)
db       = Database()
vstore   = VectorStore(settings)
llm      = LLMClient(settings)
reporter = ReportGenerator(settings)


def run_pipeline() -> tuple[str, int]:
    """Run the full ingest → summarize → export pipeline.

    Returns:
        (summary_markdown, article_count)

    Raises:
        RuntimeError on pipeline failure. Callers translate to their own
        error type (HTTPException for the API, log-and-swallow for the scheduler).
    """
    from datetime import datetime

    # 1. Fetch RSS first; World News API is a fallback when RSS yield is low.
    articles = rss.fetch_from_feeds()
    threshold = settings.RSS_YIELD_THRESHOLD
    if len(articles) < threshold:
        logger.warning(
            f"RSS yield {len(articles)} below threshold {threshold} — "
            f"falling back to World News API"
        )
        articles += news_api.fetch_financial_news()
    else:
        logger.info(
            f"RSS yield {len(articles)} meets threshold {threshold} — "
            f"skipping World News API"
        )
    if not articles:
        raise RuntimeError("No articles fetched")

    # 1b. Drop articles missing required fields
    articles = [a for a in articles if a.get("title") and a.get("content")]
    if not articles:
        raise RuntimeError("No valid articles after filtering")

    # 2. Canonicalize URLs. Drop articles whose URL can't be parsed.
    survivors = []
    for a in articles:
        try:
            a["canonical_url"] = canonicalize_url(a["url"])
            survivors.append(a)
        except ValueError as e:
            logger.warning(f"Dropping article — unparseable URL {a.get('url')!r}: {e}")
    articles = survivors

    # 3. Clean each article's content in place.
    for a in articles:
        a["content"] = cleaner.clean_article(a["content"])

    # 4. Compute content hash AFTER cleaning so ads/whitespace don't bleed in.
    survivors = []
    for a in articles:
        try:
            a["content_hash"] = compute_content_hash(a)
            survivors.append(a)
        except (KeyError, ValueError) as e:
            logger.warning(f"Dropping article — bad content {a.get('url')!r}: {e}")
    articles = survivors

    # 5. Deduplicate
    articles = dedup.deduplicate_articles(articles)

    # 4. Persist to SQL (skips existing URLs)
    db.save_articles(articles)

    # 5. Chunk + index newly-saved articles into ChromaDB (chunk-level).
    #    MUST run after save_articles — chunk ids need the SQLite autoincrement id.
    #    get_unindexed_articles() returns ORM rows; chunk_article() wants dicts.
    unindexed = db.get_unindexed_articles()
    chunks = []
    for row in unindexed:
        article = {
            "id": row.id,
            "content": row.content,
            "title": row.title,
            "source": row.source,
            "url": row.url,
            "published_at": row.published_at,
        }
        chunks.extend(chunk_article(
            article,
            embedder.model.tokenizer,
            settings.CHUNK_SIZE_TOKENS,
            settings.CHUNK_OVERLAP_TOKENS,
        ))
    if chunks:
        chunk_embeddings = embedder.generate_embeddings(
            [c["text"] for c in chunks]
        ).tolist()
        vstore.add_chunks(chunks, chunk_embeddings)
        logger.info(f"Indexed {len(chunks)} chunks from {len(unindexed)} articles")
    # Mark every fetched article indexed — even one that yielded no chunks — so
    # re-runs don't keep re-processing it. Relies on mark_indexed committing.
    indexed_ids = [row.id for row in unindexed]
    if indexed_ids:
        db.mark_indexed(indexed_ids)

    # 6. LLM summary
    summary = llm.generate_summary(articles)
    if not summary:
        raise RuntimeError("LLM returned empty summary")

    # 7. Export Markdown + HTML
    now = datetime.now()
    reporter.save_markdown(summary, now)
    reporter.generate_html(summary, now)

    # 8. Save report row in DB
    db.save_report(summary, len(articles))

    return summary, len(articles)
