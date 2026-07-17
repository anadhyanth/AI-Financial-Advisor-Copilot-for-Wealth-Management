"""
Streamlit frontend for the AI Financial Advisor Copilot.

Talks to the FastAPI backend over HTTP (never imports src/ directly), so the
frontend and backend can be deployed and scaled independently.

Run locally (with the API already running):
    streamlit run frontend/streamlit_app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Financial Advisor Copilot", page_icon="💼", layout="wide")


def call_api(method: str, endpoint: str, **kwargs):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except requests.exceptions.RequestException as e:
        return None, f"Could not reach the API at {API_BASE_URL}: {e}"


def render_sidebar_health():
    data, error = call_api("GET", "/health")
    with st.sidebar:
        st.subheader("System status")
        if error:
            st.error(f"API unreachable: {error}")
            return
        status_color = "🟢" if data["status"] == "ok" else "🟡"
        st.write(f"{status_color} API status: **{data['status']}**")
        st.write(f"Knowledge base loaded: {'✅' if data['vectorstore_loaded'] else '❌'}")
        st.write(f"LLM configured: {'✅' if data['llm_configured'] else '❌'}")


def render_chat_tab():
    st.header("Client & Advisor Chat")
    st.caption("Ask about portfolio performance, market concepts, or client documents. "
               "Answers are grounded in your firm's knowledge base with source citations.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {
                    "query": query,
                    "conversation_history": st.session_state.chat_history[:-1] or None,
                }
                data, error = call_api("POST", "/chat", json=payload)

            if error:
                st.error(error)
            else:
                st.write(data["answer"])
                if data["sources"]:
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.write(f"- {s['source']} (relevance: {s['score']})")
                st.session_state.chat_history.append({"role": "assistant", "content": data["answer"]})


def render_portfolio_tab():
    st.header("Portfolio Analysis")
    st.caption("Upload holdings (ticker, quantity, cost_basis) as a CSV, or edit the sample table below.")

    client_name = st.text_input("Client name", value="Sample Client")
    period = st.selectbox("Lookback period", ["3mo", "6mo", "1y", "3y", "5y"], index=2)

    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    if uploaded is not None:
        holdings_df = pd.read_csv(uploaded)
    else:
        holdings_df = pd.DataFrame([
            {"ticker": "AAPL", "quantity": 50, "cost_basis": 6000},
            {"ticker": "MSFT", "quantity": 30, "cost_basis": 8000},
            {"ticker": "VOO", "quantity": 20, "cost_basis": 7500},
        ])

    holdings_df = st.data_editor(holdings_df, num_rows="dynamic", use_container_width=True)

    if st.button("Analyze portfolio", type="primary"):
        holdings = holdings_df.to_dict(orient="records")
        payload = {"client_name": client_name, "holdings": holdings, "period": period}

        with st.spinner("Fetching prices and computing metrics..."):
            data, error = call_api("POST", "/portfolio/analyze", json=payload)

        if error:
            st.error(error)
        else:
            st.session_state["last_portfolio_payload"] = payload
            st.session_state["last_portfolio_result"] = data

    if "last_portfolio_result" in st.session_state:
        data = st.session_state["last_portfolio_result"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Market Value", f"${data['total_market_value']:,.2f}")
        col2.metric("Total Return", f"{data['total_return_pct']}%")
        col3.metric("Sharpe Ratio", f"{data['sharpe_ratio']}")
        col4.metric("Max Drawdown", f"{data['max_drawdown_pct']}%")

        st.subheader("Allocation")
        alloc_df = pd.DataFrame(data["allocation"])
        st.dataframe(alloc_df, use_container_width=True)
        st.bar_chart(alloc_df.set_index("ticker")["weight_pct"])


def render_report_tab():
    st.header("Client Report Generation")
    st.caption("Generates a PDF report combining portfolio metrics with an LLM-written narrative summary.")

    if "last_portfolio_payload" not in st.session_state:
        st.info("Run a portfolio analysis in the Portfolio tab first.")
        return

    payload = dict(st.session_state["last_portfolio_payload"])
    include_narrative = st.checkbox("Include AI-written narrative summary", value=True)

    if st.button("Generate PDF report", type="primary"):
        payload["include_narrative"] = include_narrative
        with st.spinner("Generating report..."):
            data, error = call_api("POST", "/report/generate", json=payload)

        if error:
            st.error(error)
        else:
            st.success("Report generated.")
            download_url = f"{API_BASE_URL}/report/download?path={data['report_path']}"
            st.markdown(f"[Download report]({download_url})")


def main():
    st.title("💼 AI Financial Advisor Copilot")
    render_sidebar_health()

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Portfolio Analysis", "📄 Reports"])
    with tab1:
        render_chat_tab()
    with tab2:
        render_portfolio_tab()
    with tab3:
        render_report_tab()


if __name__ == "__main__":
    main()
