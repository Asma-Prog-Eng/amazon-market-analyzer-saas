import os
import re
import sqlite3
from io import BytesIO, StringIO
import csv
from typing import List

import streamlit as st
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ==========================================
# 1. DATABASE SETUP & HELPER FUNCTIONS
# ==========================================
DB_NAME = "market_reports.db"

def init_db():
    """Initializes SQLite database for report history."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            product_name TEXT,
            price_range TEXT,
            pros TEXT,
            cons TEXT,
            recommendation TEXT,
            pricing_strategy TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_report_to_db(data: "MarketAnalysis"):
    """Saves a market analysis report into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (product_name, price_range, pros, cons, recommendation, pricing_strategy)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data.product_name,
        data.price_range,
        "|".join(data.pros),
        "|".join(data.cons),
        data.recommendation,
        data.pricing_strategy
    ))
    conn.commit()
    conn.close()

def fetch_history():
    """Fetches past report history from SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, product_name, price_range, recommendation FROM reports ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def clean_for_reportlab(text: str) -> str:
    """Sanitizes text for ReportLab Paragraph XML parsing."""
    if not text:
        return ""
    # Replace any form of <br> with strict ReportLab self-closing <br/>
    text = re.sub(r'<\s*br\s*/?\s*>', '<br/>', text, flags=re.IGNORECASE)
    # Strip unsupported HTML tags, keeping only safe formatting tags like b, i, u
    text = re.sub(r'</?\s*(?!b|i|u|br)\w+\b[^>]*>', '', text, flags=re.IGNORECASE)
    # Escape standalone bare ampersands
    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
    return text


# ==========================================
# 2. PYDANTIC SCHEMAS & STATE
# ==========================================
class MarketAnalysis(BaseModel):
    product_name: str = Field(description="Name or category of the analyzed product")
    price_range: str = Field(description="Observed market price range (e.g. '$25 - $45')")
    pros: List[str] = Field(description="Key strengths and advantages identified")
    cons: List[str] = Field(description="Common complaints, weaknesses, or market gaps")
    recommendation: str = Field(description="Strategic recommendation for entering or scaling in this market")
    pricing_strategy: str = Field(description="Detailed pricing strategy, margins, and tier structure")

class AgentState(BaseModel):
    search_query: str
    scraped_content: str = ""
    analysis: MarketAnalysis = None


# ==========================================
# 3. LANGGRAPH 3-NODE PIPELINE
# ==========================================
def node_scraper(state: AgentState) -> dict:
    """Node 1: Scrapes search results / web content for the query."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(state.search_query + ' amazon reviews price specs')}"
    
    try:
        res = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        snippets = [snippet.get_text(strip=True) for snippet in soup.select(".result__snippet")[:8]]
        scraped_text = "\n".join(snippets) if snippets else "No raw market text retrieved."
    except Exception as e:
        scraped_text = f"Scraper fallback: unable to fetch live web data ({str(e)})."

    return {"scraped_content": scraped_text}

def node_analyzer(state: AgentState) -> dict:
    """Node 2: Analyzes scraped content into structured Pydantic format."""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    structured_llm = llm.with_structured_output(MarketAnalysis)
    
    prompt = f"""You are an expert e-commerce market researcher.
Analyze the following market research text for query: '{state.search_query}'.

Scraped Content:
{state.scraped_content}

Provide a structured analysis containing:
1. Product name/category
2. Estimated price range
3. Top 3-5 pros/strengths
4. Top 3-5 cons/complaints/gaps
5. Strategic recommendation
6. Initial baseline pricing approach
"""
    result = structured_llm.invoke(prompt)
    return {"analysis": result}

def node_pricing_strategist(state: AgentState) -> dict:
    """Node 3: Refines and expands the pricing strategy in the analysis."""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    
    prompt = f"""You are a senior Amazon Pricing Strategist.
Review the market analysis below for product '{state.analysis.product_name}' (Price range: {state.analysis.price_range}).

Existing Pros: {', '.join(state.analysis.pros)}
Existing Cons: {', '.join(state.analysis.cons)}

Write a comprehensive, professional pricing & margin strategy. Include:
- Recommended base list price & target gross margin %
- Tiered quantity discount structure
- Seasonal or promotional discount recommendations
- Upsell / bundling recommendations to boost average order value
- Clear, bulleted quick action steps

Do not use unclosed HTML tags. Use plain text formatting or standard Markdown.
"""
    res = llm.invoke(prompt)
    state.analysis.pricing_strategy = res.content
    return {"analysis": state.analysis}

def build_workflow():
    """Builds and compiles the 3-node LangGraph execution graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("scraper", node_scraper)
    workflow.add_node("analyzer", node_analyzer)
    workflow.add_node("strategist", node_pricing_strategist)

    workflow.add_edge(START, "scraper")
    workflow.add_edge("scraper", "analyzer")
    workflow.add_edge("analyzer", "strategist")
    workflow.add_edge("strategist", END)

    return workflow.compile()


# ==========================================
# 4. EXPORT GENERATORS (PDF & CSV)
# ==========================================
def generate_pdf(data: MarketAnalysis) -> bytes:
    """Generates a downloadable PDF report with robust ReportLab XML safety."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=16, leading=20, spaceAfter=10)
    h2_style = ParagraphStyle('ReportH2', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=3)
    
    story = [
        Paragraph(clean_for_reportlab(f"Amazon Market Report: {data.product_name}"), title_style),
        Paragraph(clean_for_reportlab(f"<b>Estimated Price Range:</b> {data.price_range}"), body_style),
        Spacer(1, 8),
        Paragraph("Key Advantages (Pros):", h2_style)
    ]
    for pro in data.pros:
        story.append(Paragraph(clean_for_reportlab(f"• {pro}"), body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Key Complaints & Market Gaps (Cons):", h2_style))
    for con in data.cons:
        story.append(Paragraph(clean_for_reportlab(f"• {con}"), body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Strategic Recommendation:", h2_style))
    story.append(Paragraph(clean_for_reportlab(data.recommendation), body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Pricing & Margin Strategy:", h2_style))
    strategy_lines = data.pricing_strategy.split('\n')
    for line in strategy_lines:
        line_str = line.strip()
        if not line_str:
            continue
        # Skip markdown table header separators (e.g. |---|---|)
        if re.match(r'^[\|\-\:\s]+$', line_str):
            continue
        # Clean markdown table pipes into readable spacing if they are part of a table row
        if line_str.startswith('|') and line_str.endswith('|'):
            cells = [c.strip() for c in line_str.split('|')[1:-1] if c.strip()]
            line_str = " | ".join(cells)
            
        story.append(Paragraph(clean_for_reportlab(line_str), body_style))
        story.append(Spacer(1, 2))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_csv(data: MarketAnalysis) -> str:
    """Generates a CSV formatted string report."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Field", "Value"])
    writer.writerow(["Product Name", data.product_name])
    writer.writerow(["Price Range", data.price_range])
    writer.writerow(["Pros", " | ".join(data.pros)])
    writer.writerow(["Cons", " | ".join(data.cons)])
    writer.writerow(["Recommendation", data.recommendation])
    writer.writerow(["Pricing Strategy", data.pricing_strategy.replace("\n", " ")])
    return output.getvalue()


# ==========================================
# 5. STREAMLIT UI APPLICATION
# ==========================================
init_db()

st.set_page_config(
    page_title="Amazon Market Analysis SaaS",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Amazon Market Analysis SaaS")
st.subheader("Multi-Agent Intelligence Platform Powered by LangGraph & Groq")

# Sidebar Configuration
st.sidebar.header("🔑 Credentials & Settings")
default_key = os.environ.get("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input("Groq API Key", value=default_key, type="password")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Historical Reports Accordion in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("📜 Saved Reports History")
history_rows = fetch_history()
if history_rows:
    for row in history_rows[:5]:
        st.sidebar.caption(f"**ID #{row[0]}** | {row[1][:10]} | {row[2]}")
else:
    st.sidebar.caption("No archived reports found.")

# Main Form Input
search_query = st.text_input("Product Search Query", placeholder="e.g. Wireless Ergonomic Mouse, Mechanical Keyboard, Foam Roller")

if st.button("Run Market Analysis", type="primary"):
    if not groq_api_key:
        st.error("Please enter a valid Groq API Key in the sidebar or Streamlit secrets to proceed.")
    elif not search_query.strip():
        st.warning("Please enter a product search query.")
    else:
        with st.spinner("Executing 3-Node Agent Workflow (Scraper -> Analyzer -> Pricing Strategist)..."):
            try:
                graph = build_workflow()
                initial_state = AgentState(search_query=search_query)
                final_output = graph.invoke(initial_state)
                analysis: MarketAnalysis = final_output["analysis"]
                
                # Save into DB
                save_report_to_db(analysis)
                st.session_state["current_analysis"] = analysis
                st.success("Analysis complete and archived to database!")
                
            except Exception as e:
                st.error(f"Error executing agent workflow: {str(e)}")

# Display Results if available in session state
if "current_analysis" in st.session_state:
    data: MarketAnalysis = st.session_state["current_analysis"]
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📊 Market Report: {data.product_name}")
        st.markdown(f"**Estimated Price Range:** `{data.price_range}`")
        
        st.subheader("✅ Key Pros")
        for pro in data.pros:
            st.markdown(f"- {pro}")
            
        st.subheader("⚠️ Key Cons & Market Gaps")
        for con in data.cons:
            st.markdown(f"- {con}")
            
        st.subheader("💡 Strategic Recommendation")
        st.info(data.recommendation)
        
        st.subheader("🎯 Pricing & Margin Strategy")
        st.markdown(data.pricing_strategy)
        
    with col2:
        st.header("📥 Export Market Report")
        
        pdf_bytes = generate_pdf(data)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{data.product_name.replace(' ', '_')}_market_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        csv_data = generate_csv(data)
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_data,
            file_name=f"{data.product_name.replace(' ', '_')}_market_report.csv",
            mime="text/csv",
            use_container_width=True
        )