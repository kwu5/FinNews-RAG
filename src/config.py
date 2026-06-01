from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    

    #Only if you want to access it through the Settings class.
    NEWS_API_KEY: str
    OPENAI_API_KEY: str
    DATABASE_URL: str = "sqlite:///./data/news.db"
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_ARTICLES_PER_DAY: int = 30
    SIMILARITY_THRESHOLD: float = 0.85
    CHUNK_SIZE_TOKENS: int = 256      # all-MiniLM-L6-v2's truncation cap; eval axis in Ship H
    CHUNK_OVERLAP_TOKENS: int = 38    # ~15% of chunk size; tokens carried into the next chunk
    DAILY_RUN_HOUR: int = 18
    OUTPUT_DIR: str = "./output"
    FEEDS_CONFIG_PATH: str = "./config/feeds.yaml"
    RSS_YIELD_THRESHOLD: int = 15  # below this, fall back to World News API


