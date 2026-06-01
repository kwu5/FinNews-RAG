from typing import Dict, List
import chromadb
from src.config import Settings



class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.financial_news_collection = self.chroma_client.get_or_create_collection(name='financial_news')


    def add_chunks(self, chunks: List[Dict], embeddings: list):
        # One ChromaDB entry per chunk. id = "{article_id}:{chunk_index}" so the
        # eval harness (Ship F) can map a retrieved chunk back to its source row.
        # Metadata can't hold None, so coerce missing optional fields to "".
        ids, documents, metadatas = [], [], []
        for c in chunks:
            ids.append(f"{c['article_id']}:{c['chunk_index']}")
            documents.append(c["text"])
            metadatas.append({
                "article_id": c["article_id"],
                "chunk_index": c["chunk_index"],
                "title": c.get("title") or "",
                "source": c.get("source") or "",
                "url": c.get("url") or "",
                "published_at": c.get("published_at") or "",
            })
        self.financial_news_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return

    def search_similar(self,query_embedding: list[float], n_results=10 ):
        return self.financial_news_collection.query(query_embeddings=[query_embedding],n_results=n_results)
        
        
    
    
    


if __name__ == '__main__':
    from src.processing.embeddings import EmbeddingGenerator

    settings = Settings()  # type: ignore
    vector_store = VectorStore(settings)
    embedding_gen = EmbeddingGenerator(settings)

    # Create fake chunk dicts to test add_chunks (shape matches chunker output)
    fake_chunks = [
        {"chunk_index": 0, "article_id": 1, "title": "Fed Raises Interest Rates by 0.25%",
         "text": "The Federal Reserve raised interest rates by 25 basis points on Wednesday, signaling more hikes ahead.",
         "url": "https://example.com/fed-rates", "source": "Reuters", "published_at": "2026-04-14"},
        {"chunk_index": 0, "article_id": 2, "title": "Apple Reports Record Q2 Earnings",
         "text": "Apple Inc reported record quarterly revenue of $95 billion, driven by strong iPhone sales.",
         "url": "https://example.com/apple-earnings", "source": "CNBC", "published_at": "2026-04-14"},
        {"chunk_index": 0, "article_id": 3, "title": "Bitcoin Surges Past $70,000",
         "text": "Bitcoin surged past $70,000 for the first time as institutional investors increased their holdings.",
         "url": "https://example.com/bitcoin-surge", "source": "Yahoo Finance", "published_at": "2026-04-14"},
    ]

    # Generate embeddings (over chunk text) and add chunks
    texts = [c["text"] for c in fake_chunks]
    embeddings = embedding_gen.generate_embeddings(texts).tolist()
    vector_store.add_chunks(fake_chunks, embeddings)
    print(f"Added {len(fake_chunks)} chunks to vector store")

    # Search for similar articles using a query
    query = "cryptocurrency price rally"
    query_embedding = embedding_gen.generate_embedding(query).tolist()
    results = vector_store.search_similar(query_embedding, n_results=2)

    print(f"\nQuery: '{query}'")
    print(f"Top {len(results['ids'][0])} results:")
    for i, (doc_id, distance, metadata) in enumerate(zip(results['ids'][0], results['distances'][0], results['metadatas'][0])):
        print(f"  {i+1}. {metadata['title']} (source: {metadata['source']}, distance: {distance:.4f})")