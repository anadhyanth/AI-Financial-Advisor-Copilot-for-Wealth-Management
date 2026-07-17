# AI Financial Advisor Copilot for Wealth Management

A RAG-powered copilot for financial advisors: grounded document Q&A, live
portfolio analytics, and AI-narrated client PDF reports — served via FastAPI
with a Streamlit front end.

## Architecture

```
                     ┌─────────────────────┐
                     │  Streamlit Frontend │  (chat / portfolio / reports UI)
                     └──────────┬──────────┘
                                │ HTTP
                     ┌──────────▼──────────┐
                     │     FastAPI API      │  /chat  /portfolio/analyze  /report/generate
                     └──────────┬──────────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      RAG pipeline      Portfolio analytics   Report generator
   (FAISS + embeddings)   (yfinance + risk)   (reportlab + LLM)
              │                                     │
              └───────────────► LLM ◄───────────────┘
                          (Anthropic Claude)
```

## Project structure

```
ai-financial-advisor-copilot/
├── config.py                    # Central config: paths, LLM, RAG, risk params
├── requirements.txt              # Backend dependencies
├── requirements-frontend.txt     # Lightweight frontend-only dependencies
├── Dockerfile.api                 # FastAPI backend image
├── Dockerfile.streamlit           # Streamlit frontend image
├── docker-compose.yml
├── .env.example
├── data/
│   ├── documents/                # Knowledge base source docs (fact sheets, commentary)
│   └── portfolios/                # Sample holdings CSVs
├── vectorstore/                   # Persisted FAISS index + chunk metadata
├── reports/                       # Generated client PDF reports
├── src/
│   ├── llm_client.py              # Anthropic Messages API wrapper
│   ├── ingestion.py                # Load, chunk, embed, index documents
│   ├── rag_pipeline.py             # Retrieval + grounded answer generation
│   ├── portfolio_analytics.py      # Returns, volatility, Sharpe, drawdown, allocation
│   ├── report_generator.py         # LLM narrative + PDF report builder
│   └── utils.py                    # Logging, pickle IO
├── api/
│   ├── app.py                      # FastAPI endpoints
│   └── schemas.py                  # Pydantic request/response models
├── frontend/
│   └── streamlit_app.py            # Chat / Portfolio / Reports UI
└── tests/
    └── test_api.py
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` to a valid key from the
[Claude Console](https://console.anthropic.com/).

## 2. Build the knowledge base (RAG ingestion)

Drop your firm's fund fact sheets, market commentary, and policy documents
(`.txt`, `.md`, `.pdf`) into `data/documents/`. Two sample documents are
included so you can test the pipeline immediately.

```bash
python -m src.ingestion
```

This chunks each document, embeds chunks locally with
`sentence-transformers` (no external API calls, so it's free and keeps
client documents from leaving your infrastructure), and builds a FAISS
index in `vectorstore/`.

## 3. Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: `http://localhost:8000/docs`

### Example: chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the expense ratio of the Balanced Growth Fund?"}'
```

### Example: portfolio analysis

```bash
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "Jane Doe",
    "period": "1y",
    "holdings": [
      {"ticker": "AAPL", "quantity": 50, "cost_basis": 6000},
      {"ticker": "MSFT", "quantity": 30, "cost_basis": 8000},
      {"ticker": "VOO", "quantity": 20, "cost_basis": 7500}
    ]
  }'
```

### Example: generate a client report

```bash
curl -X POST http://localhost:8000/report/generate \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "Jane Doe",
    "period": "1y",
    "include_narrative": true,
    "holdings": [
      {"ticker": "AAPL", "quantity": 50, "cost_basis": 6000},
      {"ticker": "MSFT", "quantity": 30, "cost_basis": 8000}
    ]
  }'
```

## 4. Run the Streamlit frontend

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`. The three tabs cover conversational Q&A,
portfolio analysis with an editable holdings table, and one-click PDF
report generation.

## 5. Deploy with Docker

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build -d
```

This starts two containers:
- `advisor-api` on port 8000 (FastAPI, healthchecked)
- `advisor-frontend` on port 8501 (Streamlit, waits for the API to be healthy)

The FAISS index, reports, and data directories are mounted as volumes so
they persist across container restarts.

## 6. Tests

```bash
pytest tests/ -v
```

Tests that require a live LLM key or a built knowledge base are skipped
automatically when those prerequisites aren't present, so CI can still run
the structural/validation tests without secrets.

## Design notes

- **RAG grounding, not hallucination**: the system prompt explicitly
  instructs the model to say when the knowledge base doesn't contain an
  answer rather than guessing, and every `/chat` response returns its
  source chunks with similarity scores for auditability.
- **Local embeddings**: `sentence-transformers` runs on-device, so client
  documents used for retrieval never leave your infrastructure — only the
  final assembled prompt (retrieved text + user question) is sent to the
  LLM API.
- **Portfolio math lives outside the LLM**: returns, volatility, Sharpe
  ratio, and drawdown are computed deterministically in
  `portfolio_analytics.py`. The LLM only narrates numbers it's given; it
  never computes or invents financial figures.
- **Compliance-aware prompting**: both the chat system prompt and the
  report narrative prompt include an explicit instruction to avoid
  individualized investment recommendations and to include a
  not-personalized-advice disclaimer, and the generated PDF report carries
  a written disclaimer as well.
- **Separation of concerns**: the Streamlit app only talks to the FastAPI
  service over HTTP — it never imports `src/` directly — so the backend can
  be scaled, versioned, or replaced independently of the UI.
- **Production hardening ideas**: put the API behind authentication
  (OAuth2/JWT), add per-advisor rate limiting, swap the FAISS flat index
  for an IVF/HNSW index if the knowledge base grows large, and replace
  `yfinance` with your firm's licensed market data feed for production
  pricing accuracy.
