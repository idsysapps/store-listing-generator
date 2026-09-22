from datetime import datetime
from pydantic import BaseModel, Field


class TrendQuery(BaseModel):
    id: int | None = None
    seed_keyword: str = Field(..., max_length=255)
    query: str = Field(..., max_length=500)
    created_at: datetime | None = None


class TrendScore(BaseModel):
    id: int | None = None
    query_id: int
    score: int
    delta: int = 0
    region: str = Field(default="US", max_length=10)
    fetched_at: datetime | None = None


class TrendResult(BaseModel):
    query: str
    score: int
    delta: int
    region: str = "US"
    fetched_at: datetime


class TrendHarvestRequest(BaseModel):
    seed_keywords: list[str] = Field(
        default=["funny t-shirt", "hoodie", "gift"],
        description="Seed keywords to generate trend queries"
    )
    timeframe: str = Field(
        default="today 3-m",
        description="Timeframe for trend data (pytrends format)"
    )
    region: str = Field(
        default="US",
        description="Geographic region for trends"
    )
