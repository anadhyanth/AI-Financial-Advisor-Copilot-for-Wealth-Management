"""
Production FastAPI backend for the AI Financial Advisor Copilot.

Endpoints:
  GET  /health                -> liveness/readiness probe
  POST /chat                   -> RAG-grounded conversational Q&A
  POST /portfolio/analyze      -> compute portfolio performance & risk metrics
  POST /report/generate        -> generate an LLM-narrated PDF client report

Run locally:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config
from src.portfolio_analytics import Holding, analyze_portfolio
from src.rag_pipeline import answer_query
from src.report_generator import generate_client_report
from src.utils import get_logger
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PortfolioAnalyzeRequest,
    PortfolioAnalyzeResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
)

logger = get_logger(__name__)

app = FastAPI(
    title="AI Financial Advisor Copilot API",
    description="RAG-powered portfolio insights, client Q&A, and report generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to known frontends in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    vectorstore_loaded = config.FAISS_INDEX_PATH.exists() and config.CHUNK_STORE_PATH.exists()
    llm_configured = bool(config.ANTHROPIC_API_KEY)
    status = "ok" if (vectorstore_loaded and llm_configured) else "degraded"
    return HealthResponse(
        status=status,
        vectorstore_loaded=vectorstore_loaded,
        llm_configured=llm_configured,
    )


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(request: ChatRequest):
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="LLM is not configured (missing API key).")

    history = (
        [{"role": m.role, "content": m.content} for m in request.conversation_history]
        if request.conversation_history else None
    )
    try:
        result = answer_query(request.query, conversation_history=history)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not built yet. Run `python -m src.ingestion` first.",
        )
    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/portfolio/analyze", response_model=PortfolioAnalyzeResponse, tags=["portfolio"])
def portfolio_analyze(request: PortfolioAnalyzeRequest):
    holdings = [Holding(ticker=h.ticker.upper(), quantity=h.quantity, cost_basis=h.cost_basis)
                for h in request.holdings]
    try:
        summary = analyze_portfolio(holdings, period=request.period)
        return PortfolioAnalyzeResponse(**summary.__dict__)
    except Exception as e:
        logger.exception("Portfolio analysis failed")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {e}")


@app.post("/report/generate", response_model=ReportGenerateResponse, tags=["reporting"])
def report_generate(request: ReportGenerateRequest):
    holdings = [Holding(ticker=h.ticker.upper(), quantity=h.quantity, cost_basis=h.cost_basis)
                for h in request.holdings]
    try:
        summary = analyze_portfolio(holdings, period=request.period)
        path = generate_client_report(
            summary, client_name=request.client_name, include_narrative=request.include_narrative,
        )
        return ReportGenerateResponse(
            report_path=str(path),
            message="Report generated successfully. Fetch it via GET /report/download.",
        )
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@app.get("/report/download", tags=["reporting"])
def report_download(path: str):
    file_path = Path(path)
    if not file_path.exists() or file_path.parent != config.REPORTS_DIR:
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(file_path, media_type="application/pdf", filename=file_path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host=config.API_HOST, port=config.API_PORT, reload=False)
