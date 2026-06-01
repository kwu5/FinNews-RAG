from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, or_
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import Settings

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    content = Column(String, nullable=False)
    content_hash = Column(String, unique=True, nullable=False)
    url = Column(String, unique=True, nullable=False)
    canonical_url = Column(String, unique=True, nullable=False)
    source = Column(String)
    published_at = Column(String, nullable=False)
    fetched_at = Column(String, nullable=False)
    extraction_method = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    indexed = Column(Boolean, default=False)

_ARTICLE_COLUMNS = {c.name for c in Article.__table__.columns}


class DailyReport(Base):
    __tablename__ = "daily_reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(String,  unique=True)
    content = Column(String)
    article_count = Column(Integer)
    created_at = Column(String)
    

class Database:
    def __init__(self) -> None:
        self.engine = create_engine(Settings().DATABASE_URL)    #type: ignore
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine,expire_on_commit=False)
        
    
    #insert articles, skip if url / canonical_url / content_hash already exists
    def save_articles(self, articles:list) :
        session = self.SessionLocal()
        try:
            for a in articles:
                existing = session.query(Article).filter(
                    or_(
                        Article.url == a["url"],
                        Article.canonical_url == a["canonical_url"],
                        Article.content_hash == a["content_hash"],
                    )
                ).first()
                if existing:
                    continue
                new_article = Article(**{k: v for k, v in a.items() if k in _ARTICLE_COLUMNS})
                session.add(new_article)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error:{e}")
        finally:
            session.close()
        
    #query where processed = False
    def get_unprocessed_article(self)  -> list:
        session = self.SessionLocal()
        try:
            return session.query(Article).filter_by(processed=False).all()
        finally:
            session.close()
    
    #upsert a DailyReport keyed by today's date
    def save_report(self, content:str, article_count: int) :
        session = self.SessionLocal()
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            existing = session.query(DailyReport).filter_by(report_date=today).first()
            if existing:
                existing.content = content
                existing.article_count = article_count
                existing.created_at = datetime.now().isoformat()
            else:
                session.add(DailyReport(
                    content=content,
                    article_count=article_count,
                    created_at=datetime.now().isoformat(),
                    report_date=today,
                ))
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error:{e}")
        finally:
            session.close()
    
    #update processed = True
    def mark_articles_processed(self, article_ids:list) :
        session = self.SessionLocal()
        try:
            session.query(Article).filter(Article.id.in_(article_ids)).update({"processed":True})
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error:{e}")
        finally:
            session.close()

    #query where indexed = False — articles not yet chunked + embedded into ChromaDB
    def get_unindexed_articles(self) -> list:
        session = self.SessionLocal()
        try:
            return session.query(Article).filter_by(indexed=False).all()
        finally:
            session.close()

    #update indexed = True. Must commit — re-run safety relies on this flag persisting.
    def mark_indexed(self, article_ids:list) :
        session = self.SessionLocal()
        try:
            session.query(Article).filter(Article.id.in_(article_ids)).update({"indexed":True})
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error:{e}")
        finally:
            session.close()
    






if __name__ == '__main__':
      db = Database()

      # Test save_articles
      articles = [
          {
              "title": "Fed Raises Interest Rates by 0.25%",
              "description": "The Federal Reserve raised interest rates by a quarter point.",
              "content": "The Federal Reserve raised interest rates by a quarter point on Wednesday.",
              "url": "https://example.com/fed-rate-hike",
              "source": "Reuters",
              "published_at": "2026-03-31T10:00:00",
              "fetched_at": "2026-03-31T12:00:00",
          },
          {
              "title": "Apple Reports Record Q4 Earnings",
              "description": "Apple beat Wall Street expectations.",
              "content": "Apple beat Wall Street expectations with strong iPhone sales.",
              "url": "https://example.com/apple-earnings",
              "source": "CNBC",
              "published_at": "2026-03-31T11:00:00",
              "fetched_at": "2026-03-31T12:00:00",
          },
      ]
      db.save_articles(articles)
      print("Saved articles")

      # Test save_articles duplicate skip
      db.save_articles(articles)
      print("Duplicate insert skipped")

      # Test get_unprocessed_article
      unprocessed = db.get_unprocessed_article()
      for a in unprocessed:
          print(f"  [{a.id}] {a.title} (processed={a.processed})")

      # Test mark_articles_processed
      ids = [a.id for a in unprocessed]
      db.mark_articles_processed(ids)
      print(f"Marked {ids} as processed")

      # Verify processed
      still_unprocessed = db.get_unprocessed_article()
      print(f"Unprocessed remaining: {len(still_unprocessed)}")

      # Test save_report
      db.save_report(content="Today's market summary...", article_count=2)
      print("Saved report")



