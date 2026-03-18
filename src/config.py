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
DAILY_RUN_HOUR: int = 18  


