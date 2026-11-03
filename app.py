import os
import json
import sqlite3
from datetime import datetime
from io import BytesIO
import pandas as pd
import streamlit as st

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Import compiled LangGraph workflow & Pydantic schema from amazon_agent.py
from amazon_agent import app, MarketAnalysis

# ------------------------------------------------------------------------------
# SQLite Database Setup & Helpers
# ------------------------------------------------------------------------------
DB_FILE = "market_reports.db"

def init_db():
    """Initializes the SQLite database table for report history."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            product_name TEXT,
            price_range TEXT,
            pros TEXT,
            cons TEXT,
            recommendation TEXT,
            pricing_strategy TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_report(data: MarketAnalysis):
    """Saves a completed MarketAnalysis object into SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reports (timestamp, product_name, price_range, pros, cons, recommendation, pricing_strategy)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.product_name,
        data.price_range,
        json.dumps(data.pros),
        json.dumps(data.cons),
        data.recommendation,
        data.pricing_strategy
    ))
    conn.commit()
    conn.close()

def get_all_reports():
    """Retrieves all past reports from the database, newest first."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, product_name, price_range, pros, cons, recommendation, pricing_strategy FROM reports ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

# Initialize database on app load
init_db()


# ------------------------------------------------------------------------------
# Export Helper Functions (CSV & PDF)
# ------------------------------------------------------------------------------
def generate_csv(data: MarketAnalysis) -> bytes:
    export_dict = {
        "Metric / Key": ["Product Category", "Price Range", "Pros", "Cons", "Recommendation", "Pricing Strategy"],
        "Details": [
            data.product_name,
            data.price_range,
            "; ".join(data.pros),
            "; ".join(data.cons),
            data.recommendation,
            data.pricing_strategy
        ]
    }
    df = pd.DataFrame(export_dict)
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf(data: MarketAnalysis) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=12)
    h2_style = ParagraphStyle('ReportH2', parent=styles['Heading2'], fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=4)
    
    story = [
        Paragraph(f"Amazon Market Report: {data.product_name}", title_style),
        Paragraph(f"<b>Estimated Price Range:</b> {data.price_range}", body_style),
        Spacer(1, 12),
        Paragraph("Key Advantages (Pros):", h2_style)
    ]
    for pro in data.pros:
        story.append(Paragraph(f"• {pro}", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Key Complaints (Cons):", h2_style))
    for con in data.cons:
        story.append(Paragraph(f"• {con}", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Strategic Recommendation:", h2_style))
    story.append(Paragraph(data.recommendation, body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Pricing & Margin Strategy:", h2_style))
    story.append(Paragraph(data.pricing_strategy, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ------------------------------------------------------------------------------
# Streamlit Web Application Interface
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Market Analyzer SaaS",
    page_icon="📦",
    layout="wide"
)

# Sidebar Configuration & Navigation
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
    
    st.markdown("---")
    nav_choice = st.radio("Navigation", ["🔍 Run Analysis", "📜 Saved Report History"])
    
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("Powered by LangGraph (3 Nodes), Pydantic, SQLite, and Streamlit.")


# --- VIEW 1: RUN LIVE ANALYSIS ---
if nav_choice == "🔍 Run Analysis":
    st.title("📦 Amazon Agentic Market Analysis")
    st.caption("Multi-agent workflow: BeautifulSoup Scraper ➔ Pydantic Analyzer ➔ Pricing Strategist")

    query = st.text_input("Enter Amazon Product / Category:", placeholder="e.g., wireless ergonomic mouse")

    if st.button("Run Market Analysis", type="primary"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please enter or set your GROQ_API_KEY in the sidebar to proceed.")
        elif not query.strip():
            st.warning("Please enter a valid product search query.")
        else:
            with st.spinner("Executing 3-node agent workflow..."):
                try:
                    result = app.invoke({"query": query})
                    data: MarketAnalysis = result["analysis"]

                    # Automatically save report to SQLite database
                    save_report(data)

                    st.success("Analysis Complete & Saved to Database!")
                    st.divider()

                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Product Category", data.product_name)
                    with col2:
                        st.metric("Estimated Price Range", data.price_range)

                    st.divider()

                    # Pros & Cons
                    p_col, c_col = st.columns(2)
                    with p_col:
                        st.subheader("👍 Key Pros")
                        for pro in data.pros:
                            st.markdown(f"- {pro}")
                    with c_col:
                        st.subheader("👎 Key Cons")
                        for con in data.cons:
                            st.markdown(f"- {con}")

                    st.divider()

                    # Recommendations & Pricing Strategy
                    st.subheader("💡 Strategic Recommendation")
                    st.info(data.recommendation)

                    st.subheader("💰 Automated Pricing & Competitive Strategy")
                    st.success(data.pricing_strategy)

                    # Export Buttons
                    st.divider()
                    st.subheader("📥 Export Market Report")
                    col_csv, col_pdf = st.columns(2)
                    filename_prefix = data.product_name.lower().replace(" ", "_")

                    with col_csv:
                        st.download_button(
                            label="📄 Download Report (CSV)",
                            data=generate_csv(data),
                            file_name=f"{filename_prefix}_analysis.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_pdf:
                        st.download_button(
                            label="📕 Download Report (PDF)",
                            data=generate_pdf(data),
                            file_name=f"{filename_prefix}_analysis.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error executing agent workflow: {e}")


# --- VIEW 2: REPORT HISTORY ARCHIVE ---
elif nav_choice == "📜 Saved Report History":
    st.title("📜 Past Market Analysis Reports")
    st.caption("Review and reload past analyses saved locally in SQLite.")

    reports = get_all_reports()

    if not reports:
        st.info("No saved reports found yet. Run an analysis from the sidebar to populate history!")
    else:
        st.write(f"Found **{len(reports)}** saved report(s) in local database (`market_reports.db`).")
        st.divider()

        for row in reports:
            rep_id, timestamp, product_name, price_range, pros_json, cons_json, recommendation, pricing_strategy = row
            pros = json.loads(pros_json)
            cons = json.loads(cons_json)

            with st.expander(f"📦 [{timestamp}] {product_name} ({price_range})"):
                st.markdown(f"**Estimated Price Range:** {price_range}")
                
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown("**Pros:**")
                    for p in pros:
                        st.markdown(f"- {p}")
                with col_h2:
                    st.markdown("**Cons:**")
                    for c in cons:
                        st.markdown(f"- {c}")

                st.markdown("**Strategic Recommendation:**")
                st.info(recommendation)

                st.markdown("**Pricing Strategy:**")
                st.success(pricing_strategy)