"""
API tests. Tests that don't require external services (LLM API, market data)
always run; others are skipped gracefully when prerequisites are missing.

Run:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

import config
from api.app import app

client = TestClient(app)

KB_READY = config.FAISS_INDEX_PATH.exists() and config.CHUNK_STORE_PATH.exists()
LLM_READY = bool(config.ANTHROPIC_API_KEY)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "vectorstore_loaded" in body
    assert "llm_configured" in body


def test_chat_without_llm_key_returns_503(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    response = client.post("/chat", json={"query": "What is the fund's expense ratio?"})
    assert response.status_code == 503


@pytest.mark.skipif(not (KB_READY and LLM_READY), reason="Knowledge base or LLM API key not configured")
def test_chat_returns_grounded_answer():
    response = client.post("/chat", json={"query": "What is the fund's annual expense ratio?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert isinstance(body["sources"], list)


def test_portfolio_analyze_missing_ticker_field():
    payload = {"client_name": "Test", "holdings": [{"ticker": "AAPL", "quantity": 10}]}  # missing cost_basis
    response = client.post("/portfolio/analyze", json=payload)
    assert response.status_code == 422


def test_report_download_rejects_path_outside_reports_dir():
    response = client.get("/report/download", params={"path": "/etc/passwd"})
    assert response.status_code == 404
