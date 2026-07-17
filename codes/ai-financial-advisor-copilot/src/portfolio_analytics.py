"""
Portfolio analytics engine.

Takes a client's holdings (ticker, quantity, cost basis) plus historical
price data and computes the metrics an advisor needs for client reporting:
total value, allocation by holding, period return, annualized volatility,
Sharpe ratio, and max drawdown.

Historical prices are pulled via yfinance by default; swap `fetch_price_history`
for your firm's internal market data feed in production.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

import config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Holding:
    ticker: str
    quantity: float
    cost_basis: float  # total amount originally paid


@dataclass
class PortfolioSummary:
    total_market_value: float
    total_cost_basis: float
    total_unrealized_pnl: float
    total_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    allocation: List[dict] = field(default_factory=list)


def load_portfolio_csv(path) -> List[Holding]:
    """Expects columns: ticker, quantity, cost_basis"""
    df = pd.read_csv(path)
    required = {"ticker", "quantity", "cost_basis"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Portfolio CSV missing required columns: {missing}")

    df.columns = [c.lower() for c in df.columns]
    return [
        Holding(ticker=row.ticker.upper(), quantity=row.quantity, cost_basis=row.cost_basis)
        for row in df.itertuples()
    ]


def fetch_price_history(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Returns a DataFrame of daily adjusted close prices, columns=tickers."""
    logger.info("Fetching %s price history for: %s", period, tickers)
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna(how="all")


def compute_current_prices(price_history: pd.DataFrame) -> dict:
    return price_history.iloc[-1].to_dict()


def compute_portfolio_returns_series(price_history: pd.DataFrame, holdings: List[Holding]) -> pd.Series:
    """Daily portfolio value series, weighted by each holding's quantity."""
    weights = {h.ticker: h.quantity for h in holdings}
    value_series = sum(
        price_history[ticker] * qty
        for ticker, qty in weights.items()
        if ticker in price_history.columns
    )
    return value_series.dropna()


def compute_risk_metrics(value_series: pd.Series) -> dict:
    daily_returns = value_series.pct_change().dropna()
    if daily_returns.empty:
        return {"annualized_volatility_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}

    annualized_vol = daily_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    mean_daily_return = daily_returns.mean()
    annualized_return = mean_daily_return * config.TRADING_DAYS_PER_YEAR

    excess_return = annualized_return - config.RISK_FREE_RATE_ANNUAL
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0.0

    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "annualized_volatility_pct": round(float(annualized_vol) * 100, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
    }


def build_allocation_table(holdings: List[Holding], current_prices: dict, total_value: float) -> List[dict]:
    allocation = []
    for h in holdings:
        price = current_prices.get(h.ticker)
        if price is None:
            logger.warning("No current price found for %s, skipping in allocation", h.ticker)
            continue
        market_value = price * h.quantity
        allocation.append({
            "ticker": h.ticker,
            "quantity": h.quantity,
            "current_price": round(float(price), 2),
            "market_value": round(float(market_value), 2),
            "weight_pct": round(float(market_value / total_value) * 100, 2) if total_value else 0.0,
            "cost_basis": round(float(h.cost_basis), 2),
            "unrealized_pnl": round(float(market_value - h.cost_basis), 2),
        })
    return sorted(allocation, key=lambda x: x["weight_pct"], reverse=True)


def analyze_portfolio(holdings: List[Holding], period: str = "1y") -> PortfolioSummary:
    tickers = [h.ticker for h in holdings]
    price_history = fetch_price_history(tickers, period=period)
    current_prices = compute_current_prices(price_history)

    total_market_value = sum(
        current_prices.get(h.ticker, 0) * h.quantity for h in holdings
    )
    total_cost_basis = sum(h.cost_basis for h in holdings)
    total_pnl = total_market_value - total_cost_basis
    total_return_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis else 0.0

    value_series = compute_portfolio_returns_series(price_history, holdings)
    risk_metrics = compute_risk_metrics(value_series)

    allocation = build_allocation_table(holdings, current_prices, total_market_value)

    return PortfolioSummary(
        total_market_value=round(float(total_market_value), 2),
        total_cost_basis=round(float(total_cost_basis), 2),
        total_unrealized_pnl=round(float(total_pnl), 2),
        total_return_pct=round(float(total_return_pct), 2),
        annualized_volatility_pct=risk_metrics["annualized_volatility_pct"],
        sharpe_ratio=risk_metrics["sharpe_ratio"],
        max_drawdown_pct=risk_metrics["max_drawdown_pct"],
        allocation=allocation,
    )
