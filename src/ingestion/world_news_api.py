import logging
from typing import Optional
from pydantic import BaseModel, Field
from datetime import date
import requests

from src.config import Settings

logger = logging.getLogger(__name__)



class SearchParams(BaseModel):
    
    
    text: str = "stock+market"
    text_match_indexes: Optional[str] = Field(default=None, alias="text-match-indexes")
    source_country: Optional[str] = Field(default="us", alias="source-country")
    language: str = "en"
    min_sentiment: Optional[float] = Field(default=None, alias="min-sentiment")
    max_sentiment: Optional[float] = Field(default=None, alias="max-sentiment")
    earliest_publish_date: str = Field( default_factory=lambda: date.today().isoformat(), alias="earliest-publish-date")
    latest_publish_date: Optional[str] = Field(default=None, alias="latest-publish-date")
    news_sources: Optional[str] = Field(default=None, alias="news-sources")
    authors: Optional[str] = None
    categories: Optional[str] = None
    entities: Optional[str] = None
    location_filter: Optional[str] = Field(default=None, alias="location-filter")
    sort: Optional[str] = None
    sort_direction: Optional[str] = Field(default=None, alias="sort-direction")
    offset: Optional[int] = None
    number: int = 10


class WorldNewsAPIClient:
    def __init__(self) -> None:
        self.settings = Settings() # type: ignore
        self.base_url = "https://api.worldnewsapi.com/search-news?"
        self.headers = {'x-api-key': self.settings.NEWS_API_KEY}
        

    def fetch_financial_news(self, days_back : int = 1) -> list[dict]:
        
        
        
        
        params = SearchParams()
        query = params.model_dump(by_alias=True,exclude_none=True)
        # print(params)
        
        response = requests.get(self.base_url, params=query, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
        

if __name__ == "__main__":
    client = WorldNewsAPIClient()
    result = client.fetch_financial_news()
    print(result)
    
        
    
    
    
    
    
    
    
