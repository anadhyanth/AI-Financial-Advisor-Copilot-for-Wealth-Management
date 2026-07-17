"""
Generates a client-facing PDF report: portfolio summary table, allocation
breakdown, risk metrics, and an LLM-written plain-English narrative that
explains performance in context.
"""

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

import config
from src.llm_client import generate_response
from src.portfolio_analytics import PortfolioSummary
from src.utils import get_logger

logger = get_logger(__name__)


NARRATIVE_SYSTEM_PROMPT = """You write brief, plain-English quarterly portfolio \
summaries for wealth management clients. Use the provided metrics only — do \
not invent figures. Keep it to 3-4 short paragraphs: overall performance, \
notable allocation points, risk profile, and a closing note that this is an \
informational summary and not personalized investment advice. Avoid jargon \
where a simpler phrase works."""


def generate_narrative(summary: PortfolioSummary, client_name: str) -> str:
    metrics_text = (
        f"Total market value: ${summary.total_market_value:,.2f}\n"
        f"Total cost basis: ${summary.total_cost_basis:,.2f}\n"
        f"Unrealized P&L: ${summary.total_unrealized_pnl:,.2f}\n"
        f"Total return: {summary.total_return_pct}%\n"
        f"Annualized volatility: {summary.annualized_volatility_pct}%\n"
        f"Sharpe ratio: {summary.sharpe_ratio}\n"
        f"Max drawdown: {summary.max_drawdown_pct}%\n"
        f"Top holdings: "
        + ", ".join(f"{a['ticker']} ({a['weight_pct']}%)" for a in summary.allocation[:5])
    )

    prompt = f"Client name: {client_name}\n\nPortfolio metrics:\n{metrics_text}"
    return generate_response(system_prompt=NARRATIVE_SYSTEM_PROMPT, user_message=prompt)


def build_pdf_report(
    summary: PortfolioSummary,
    client_name: str,
    output_path: Path,
    narrative: str | None = None,
) -> Path:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=18, spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
    )
    body_style = styles["BodyText"]

    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    story.append(Paragraph("Portfolio Performance Report", title_style))
    story.append(Paragraph(f"Prepared for: {client_name}", body_style))
    story.append(Paragraph(f"Report date: {date.today().isoformat()}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Summary", heading_style))
    summary_rows = [
        ["Metric", "Value"],
        ["Total Market Value", f"${summary.total_market_value:,.2f}"],
        ["Total Cost Basis", f"${summary.total_cost_basis:,.2f}"],
        ["Unrealized P&L", f"${summary.total_unrealized_pnl:,.2f}"],
        ["Total Return", f"{summary.total_return_pct}%"],
        ["Annualized Volatility", f"{summary.annualized_volatility_pct}%"],
        ["Sharpe Ratio", f"{summary.sharpe_ratio}"],
        ["Max Drawdown", f"{summary.max_drawdown_pct}%"],
    ]
    summary_table = Table(summary_rows, colWidths=[2.5 * inch, 2.5 * inch])
    summary_table.setStyle(_default_table_style())
    story.append(summary_table)

    story.append(Paragraph("Holdings Allocation", heading_style))
    alloc_header = ["Ticker", "Qty", "Price", "Market Value", "Weight %", "Unrealized P&L"]
    alloc_rows = [alloc_header] + [
        [a["ticker"], a["quantity"], f"${a['current_price']:,.2f}",
         f"${a['market_value']:,.2f}", f"{a['weight_pct']}%", f"${a['unrealized_pnl']:,.2f}"]
        for a in summary.allocation
    ]
    alloc_table = Table(alloc_rows, colWidths=[0.8*inch, 0.6*inch, 0.8*inch, 1.1*inch, 0.8*inch, 1.1*inch])
    alloc_table.setStyle(_default_table_style())
    story.append(alloc_table)

    if narrative:
        story.append(Paragraph("Advisor Commentary", heading_style))
        for paragraph in narrative.split("\n\n"):
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), body_style))
                story.append(Spacer(1, 0.1 * inch))

    disclaimer = (
        "This report is generated for informational purposes only and does not "
        "constitute personalized investment advice. Past performance is not "
        "indicative of future results. Please consult your financial advisor "
        "before making investment decisions."
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(disclaimer, ParagraphStyle(
        "Disclaimer", parent=body_style, fontSize=8, textColor=colors.grey,
    )))

    doc.build(story)
    logger.info("Saved PDF report to %s", output_path)
    return output_path


def _default_table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ])


def generate_client_report(
    summary: PortfolioSummary,
    client_name: str,
    include_narrative: bool = True,
) -> Path:
    narrative = generate_narrative(summary, client_name) if include_narrative else None
    filename = f"{client_name.replace(' ', '_').lower()}_portfolio_report_{date.today().isoformat()}.pdf"
    output_path = config.REPORTS_DIR / filename
    return build_pdf_report(summary, client_name, output_path, narrative)
