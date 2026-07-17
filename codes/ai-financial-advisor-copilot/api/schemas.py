"""Pydantic request/response schemas for the FastAPI backend."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[ChatMessage]] = None
    client_name: Optional[str] = None


class SourceCitation(BaseModel):
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]


class HoldingInput(BaseModel):
    ticker: str
    quantity: float
    cost_basis: float


class PortfolioAnalyzeRequest(BaseModel):
    client_name: str
    holdings: List[HoldingInput]
    period: str = Field("1y", description="yfinance period string, e.g. '6mo', '1y', '5y'")


class AllocationRow(BaseModel):
    ticker: str
    quantity: float
    current_price: float
    market_value: float
    weight_pct: float
    cost_basis: float
    unrealized_pnl: float


class PortfolioAnalyzeResponse(BaseModel):
    total_market_value: float
    total_cost_basis: float
    total_unrealized_pnl: float
    total_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    allocation: List[AllocationRow]


class ReportGenerateRequest(BaseModel):
    client_name: str
    holdings: List[HoldingInput]
    period: str = "1y"
    include_narrative: bool = True


class ReportGenerateResponse(BaseModel):
    report_path: str
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    llm_configured: bool
