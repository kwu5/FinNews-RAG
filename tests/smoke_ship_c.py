"""Ship C smoke test — verify chunk-level ChromaDB indexing + the indexed gate.

Prerequisites — do these once before running:
    Remove-Item -Recurse -Force .\\data\\chroma     # task 6: wipe stale article-level index

Then run:
    python -m tests.smoke_ship_c

What it checks:
  1. STRUCTURE — after a populated run, ChromaDB holds chunk entries:
       - chunk count >= article count (expect strictly > for real long articles)
       - every id matches "{int}:{int}"
       - every chunk metadata carries article_id, and it matches the id prefix
  2. RETRIEVAL — a sample query returns passage-sized hits (eyeball the snippets).
  3. THE GATE — a second pipeline run does NOT re-chunk already-indexed articles:
       - every baseline chunk id survives unchanged
       - baseline articles' chunk sets are identical before/after
       - no article is left indexed=False

NOTE: this runs the full pipeline up to twice — real RSS fetches + OpenAI calls.
New RSS articles appearing between run 1 and run 2 are expected and fine; the gate
check only asserts that the *baseline* articles aren't reprocessed.
"""

import logging
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.pipeline import run_pipeline, db, vstore, embedder
from src.storage.database import Article

ID_RE = re.compile(r"^\d+:\d+$")


def chroma_snapshot():
    """Return (ids, metadatas) for every entry currently in the collection."""
    got = vstore.financial_news_collection.get()
    return got["ids"], got["metadatas"]


def main() -> None:
    coll = vstore.financial_news_collection
    failures = []

    # Populate if the collection is empty (e.g. right after the task-6 wipe).
    if coll.count() == 0:
        print("Chroma empty — running pipeline once to populate...\n")
        run_pipeline()

    ids, metas = chroma_snapshot()
    chunk_count = len(ids)
    with db.SessionLocal() as s:
        article_count = s.query(Article).count()

    print(f"\nSQLite articles: {article_count}")
    print(f"Chroma chunks:   {chunk_count}")

    # --- Check 1: counts ---
    if chunk_count < article_count:
        failures.append(f"chunk count {chunk_count} < article count {article_count}")
    elif chunk_count == article_count:
        print("NOTE: chunk count == article count — expected > for long articles; "
              "ok only if every article fit in a single chunk.")

    # --- Check 2: id shape ---
    bad_ids = [i for i in ids if not ID_RE.match(i)]
    if bad_ids:
        failures.append(f"{len(bad_ids)} ids not in '{{int}}:{{int}}' form, e.g. {bad_ids[:3]}")

    # --- Check 3: article_id present and consistent with the id prefix ---
    missing = [i for i, m in zip(ids, metas) if "article_id" not in (m or {})]
    if missing:
        failures.append(f"{len(missing)} chunks missing article_id metadata")
    mismatched = [
        i for i, m in zip(ids, metas)
        if m and str(m.get("article_id")) != i.split(":")[0]
    ]
    if mismatched:
        failures.append(f"{len(mismatched)} chunks: id prefix != metadata article_id")

    # --- Eyeball: sample query returns passage-sized hits ---
    print("\n--- Sample query: 'federal reserve interest rates' ---")
    q = embedder.generate_embedding("federal reserve interest rates").tolist()
    res = vstore.search_similar(q, n_results=3)
    for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
        snippet = doc[:160].replace("\n", " ")
        print(f"  [{cid}] {str(meta.get('title', ''))[:50]!r}")
        print(f"        {snippet}...  ({len(doc.split())} words)")

    # --- Check 4: the indexed gate holds across a second run ---
    print("\n--- Running pipeline again (run 2) to test the indexed gate ---\n")
    baseline = set(ids)
    run_pipeline()
    ids2, _ = chroma_snapshot()
    after = set(ids2)

    lost = baseline - after
    if lost:
        failures.append(f"{len(lost)} baseline chunk ids vanished after run 2")

    baseline_articles = {i.split(":")[0] for i in baseline}
    after_for_baseline = {i for i in after if i.split(":")[0] in baseline_articles}
    baseline_for_baseline = {i for i in baseline if i.split(":")[0] in baseline_articles}
    if after_for_baseline != baseline_for_baseline:
        failures.append("baseline articles changed chunk set after run 2 (gate leaked)")

    with db.SessionLocal() as s:
        unindexed = s.query(Article).filter_by(indexed=False).count()
    if unindexed:
        failures.append(f"{unindexed} articles still indexed=False after run 2")

    print(f"Chroma chunks after run 2: {len(after)} (was {len(baseline)})")
    print(f"Unindexed rows remaining:  {unindexed}")

    # --- Result ---
    print("\n=== RESULT ===")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASS — chunk-level index verified; the indexed gate holds across runs.")


if __name__ == "__main__":
    main()
