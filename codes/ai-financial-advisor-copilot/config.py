"""
Central configuration for the AI Financial Advisor Copilot.
All paths, model names, and RAG parameters live here so the ingestion
pipeline, RAG engine, portfolio analytics, API, and Streamlit frontend
all stay in sync.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PORTFOLIOS_DIR = DATA_DIR / "portfolios"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
REPORTS_DIR = BASE_DIR / "reports"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
CHUNK_STORE_PATH = VECTORSTORE_DIR / "chunks.pkl"

for d in (DATA_DIR, DOCUMENTS_DIR, PORTFOLIOS_DIR, VECTORSTORE_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# Set ANTHROPIC_API_KEY in your environment / .env file. Never hardcode keys.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

# ---------------------------------------------------------------------------
# RAG / embedding settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 4))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", 0.25))

SYSTEM_PROMPT = """You are a wealth management assistant helping a financial \
advisor and their clients understand portfolio performance and financial \
concepts. Ground every factual claim in the provided context or portfolio \
data. If the context does not contain the answer, say so explicitly rather \
than guessing. Never present speculation as certainty, never give \
individualized buy/sell recommendations, and remind users that this is \
informational content, not personalized financial advice, when the query \
implies a decision. Be precise, concise, and cite which document a fact \
came from when relevant."""

# ---------------------------------------------------------------------------
# Portfolio analytics settings
# ---------------------------------------------------------------------------
RISK_FREE_RATE_ANNUAL = float(os.getenv("RISK_FREE_RATE_ANNUAL", 0.04))
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# API / deployment
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
