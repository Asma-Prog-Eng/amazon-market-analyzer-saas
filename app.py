import os
import re
import psycopg2
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ==========================================
# 1. DATABASE SETUP & HELPER FUNCTIONS (SUPABASE)
# ==========================================
DB_URL = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))

def init_db():
    """Initializes PostgreSQL database table and safely adds missing columns if needed."""
    if not DB_URL:
        return
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                product_name TEXT,
                price_range TEXT,
                pros TEXT,
                cons TEXT,
                recommendation TEXT,
                pricing_strategy TEXT
            )
        """)
        cursor.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS viability_score INT;")
        cursor.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS is_winning_product BOOLEAN;")
        cursor.execute("ALTER TABLE reports ADD COLUMN IF NOT EXISTS sourcing_info TEXT;")
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database initialization error: {e}")

def save_report_to_db(data: "MarketAnalysis"):
    """Saves a market analysis report and sourcing metrics into the cloud PostgreSQL database."""
    if not DB_URL:
        return
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (product_name, price_range, pros, cons, recommendation, pricing_strategy, viability_score, is_winning_product, sourcing_info)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data.product_name,
            data.price_range,
            "|".join(data.pros),
            "|".join(data.cons),
            data.recommendation,
            data.pricing_strategy,
            data.viability_score,
            data.is_winning_product,
            data.sourcing_info
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error saving report to DB: {e}")

def fetch_history():
    """Fetches past report history from cloud PostgreSQL database."""
    if not DB_URL:
        return []
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, product_name, price_range, viability_score, is_winning_product FROM reports ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        return []

def clean_for_reportlab(text: str) -> str:
    """Sanitizes text, strips LaTeX math notation, and normalizes characters for ReportLab PDF compatibility."""
    if not text:
        return ""
    
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    text = text.replace(r'\pm', '+/-')
    text = text.replace(r'\%', '%')
    text = text.replace('\\', '')
    
    dash_variants = ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212', '\u00ad']
    for char in dash_variants:
        text = text.replace(char, '-')
        
    bullet_variants = ['\u2022', '\u2023', '\u25aa', '\u25ab', '\u25cf', '\u25cb', '\u25a0', '\u25a1']
    for char in bullet_variants:
        text = text.replace(char, '-')

    tilde_variants = ['\u223c', '\u2248', '\u02dc', '\u223d', '\u2243', '\u2245', '\u2249', '~']
    for char in tilde_variants:
        text = text.replace(char, 'approx. ')
        
    for char in ['\u2018', '\u2019', '\u201a', '\u201b']:
        text = text.replace(char, "'")
    for char in ['\u201c', '\u201d', '\u201e', '\u201f']:
        text = text.replace(char, '"')
        
    text = text.replace('\u00a0', ' ').replace('\u202f', ' ')

    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'<\s*br\s*/?\s*>', '<br/>', text, flags=re.IGNORECASE)
    text = re.sub(r'</?\s*(?!b|i|u|br)\w+\b[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', text)
    return text

def clean_for_csv(text: str) -> str:
    """Strips markdown tables, formatting tags, and uses vertical newlines for clean spreadsheet viewing."""
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if re.match(r'^[\|\-\:\s]+$', line_str):
            continue
        if line_str.startswith('|') and line_str.endswith('|'):
            cells = [c.strip() for c in line_str.split('|')[1:-1] if c.strip()]
            line_str = " -> ".join(cells)
            
        line_str = re.sub(r'\*\*(.*?)\*\*', r'\1', line_str)
        line_str = re.sub(r'\*(.*?)\*', r'\1', line_str)
        line_str = re.sub(r'^#+\s*', '', line_str)
        line_str = re.sub(r'<[^>]+>', ' ', line_str)
        
        cleaned_lines.append(line_str)
        
    return "\n".join(cleaned_lines)


# ==========================================
# 2. PYDANTIC SCHEMAS & STATE
# ==========================================
class MarketAnalysis(BaseModel):
    product_name: str = Field(description="Name or category of the analyzed product")
    price_range: str = Field(description="Observed market price range with local currency symbol")
    pros: List[str] = Field(description="Key strengths and advantages identified")
    cons: List[str] = Field(description="Common complaints, weaknesses, or market gaps")
    recommendation: str = Field(description="Strategic recommendation for entering or scaling in this market")
    pricing_strategy: str = Field(description="Detailed pricing strategy, margins, and tier structure")
    viability_score: int = Field(description="Calculated score from 1 to 100 assessing overall market viability and winning product potential")
    is_winning_product: bool = Field(description="True if the product has strong profit margins, high demand, and fixable consumer complaints; false otherwise")
    sourcing_info: str = Field(default="", description="Manufacturing upgrades, supplier vetting instructions, and compliance requirements")

class AgentState(BaseModel):
    search_query: str
    target_market: str = "US (Amazon.com)"
    scraped_content: str = ""
    analysis: MarketAnalysis = None


# ==========================================
# 3. LANGGRAPH 4-NODE PIPELINE
# ==========================================
def node_scraper(state: AgentState) -> dict:
    """Node 1: Scrapes search results tailored to the selected target market domain."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    market_domain = "amazon.com"
    if "Germany" in state.target_market:
        market_domain = "amazon.de"
    elif "Emirates" in state.target_market:
        market_domain = "amazon.ae"
    elif "Saudi Arabia" in state.target_market:
        market_domain = "amazon.sa"

    search_query_str = f"{state.search_query} {market_domain} reviews price specs best seller"
    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(search_query_str)}"
    
    try:
        res = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        snippets = [snippet.get_text(strip=True) for snippet in soup.select(".result__snippet")[:8]]
        scraped_text = "\n".join(snippets) if snippets else "No raw market text retrieved."
    except Exception as e:
        scraped_text = f"Scraper fallback: unable to fetch live web data ({str(e)})."

    return {"scraped_content": scraped_text}

def node_analyzer(state: AgentState) -> dict:
    """Node 2: Analyzes scraped content with specific regional currency and market constraints."""
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2)
    structured_llm = llm.with_structured_output(MarketAnalysis)
    
    currency_map = {
        "US (Amazon.com)": "USD ($)",
        "Germany (Amazon.de)": "EUR (€)",
        "Emirates (Amazon.ae)": "AED (AED)",
        "Saudi Arabia (Amazon.sa)": "SAR (SAR)"
    }
    local_currency = currency_map.get(state.target_market, "USD ($)")

    prompt = f"""You are an expert e-commerce market researcher and winning product hunter specialized in the **{state.target_market}** market.
Analyze the following market research text for query: '{state.search_query}'.

Scraped Content:
{state.scraped_content}

CRITICAL INSTRUCTIONS:
1. If the query is broad, select the single most prominent, trending product category or item within that space to analyze deeply.
2. All price ranges and financial figures MUST use the local currency: {local_currency}.
3. Tailor the analysis specifically to consumer behavior, competition levels, and pricing expectations in {state.target_market}.

Provide a structured analysis containing:
1. Product name/category (specific item)
2. Estimated price range (using {local_currency})
3. Top 3-5 pros/strengths
4. Top 3-5 cons/complaints/gaps
5. Strategic recommendation for {state.target_market}
6. Initial baseline pricing approach
7. Viability score (1-100 scale)
8. Boolean flag determining if this is a high-potential winning product
"""
    result = structured_llm.invoke(prompt)
    return {"analysis": result}

def node_pricing_strategist(state: AgentState) -> dict:
    """Node 3: Refines and expands the pricing strategy for the target region."""
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.3)
    
    prompt = f"""You are a senior Amazon Pricing Strategist for the **{state.target_market}** marketplace.
Review the market analysis below for product '{state.analysis.product_name}' (Price range: {state.analysis.price_range}, Viability Score: {state.analysis.viability_score}/100).

Existing Pros: {', '.join(state.analysis.pros)}
Existing Cons: {', '.join(state.analysis.cons)}

Write a comprehensive pricing strategy suited for {state.target_market} (including local VAT/tax considerations, FBA fulfillment estimates, and competitive tiering). Format key data points into a clean Markdown table with columns: | Element | Recommendation | Rationale |.
Follow the table with a few short bullet points for Quick Action Steps. Avoid LaTeX math symbols.
"""
    res = llm.invoke(prompt)
    state.analysis.pricing_strategy = res.content
    return {"analysis": state.analysis}

def node_sourcing_finder(state: AgentState) -> dict:
    """Node 4: Generates manufacturing upgrades and compliance advice tailored for the target market."""
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.3)
    
    cons_text = ', '.join(state.analysis.cons)
    prompt = f"""You are an expert Global Sourcing & Supply Chain Manager supplying e-commerce sellers targeting **{state.target_market}**.
Target Product: '{state.analysis.product_name}'
Identified Customer Complaints/Cons to Fix: {cons_text}

Provide an actionable Sourcing & Manufacturing Blueprint containing:
1. **Supplier Quality Upgrades**: Specific engineering or material instructions to give Chinese suppliers to fix the identified consumer complaints.
2. **Import & Compliance Requirements**: Essential certifications for importing into {state.target_market} (e.g., SASO/Saber certification and Arabic labeling for Saudi Arabia, CE marks for Europe, local voltage/plug standards).
3. **Vetting Checklist**: Key questions to ask suppliers regarding Trade Assurance and sample testing.

Format your response cleanly with markdown headers and bullet points. Do not include raw HTML link tags.
"""
    res = llm.invoke(prompt)
    state.analysis.sourcing_info = res.content
    return {"analysis": state.analysis}

def build_workflow():
    """Builds and compiles the 4-node LangGraph execution graph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("scraper", node_scraper)
    workflow.add_node("analyzer", node_analyzer)
    workflow.add_node("strategist", node_pricing_strategist)
    workflow.add_node("sourcing", node_sourcing_finder)

    workflow.add_edge(START, "scraper")
    workflow.add_edge("scraper", "analyzer")
    workflow.add_edge("analyzer", "strategist")
    workflow.add_edge("strategist", "sourcing")
    workflow.add_edge("sourcing", END)

    return workflow.compile()


# ==========================================
# 4. EXPORT GENERATORS (PDF & CSV)
# ==========================================
def generate_pdf(data: MarketAnalysis, target_market: str) -> bytes:
    """Generates a structured, clean PDF report with native ReportLab tables."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=16, leading=20, spaceAfter=10)
    h2_style = ParagraphStyle('ReportH2', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=3)
    table_cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontSize=8, leading=11)
    table_header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontSize=8, leading=11, fontName="Helvetica-Bold", textColor=colors.whitesmoke)
    
    status_text = "WINNING PRODUCT CANDIDATE 🚀" if data.is_winning_product else "Standard Niche"
    
    story = [
        Paragraph(clean_for_reportlab(f"Amazon Market Report ({target_market}): {data.product_name}"), title_style),
        Paragraph(clean_for_reportlab(f"<b>Viability Score:</b> {data.viability_score}/100 ({status_text})"), body_style),
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
    for line in data.pricing_strategy.split('\n'):
        if line.strip():
            story.append(Paragraph(clean_for_reportlab(line), body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Manufacturing & Compliance Blueprint:", h2_style))
    for line in data.sourcing_info.split('\n'):
        if line.strip():
            story.append(Paragraph(clean_for_reportlab(line), body_style))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_csv(data: MarketAnalysis, target_market: str) -> bytes:
    """Generates a clean, semicolon-delimited CSV bytes object with UTF-8 BOM for Excel compatibility."""
    output = StringIO()
    writer = csv.writer(output, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(["Report Section", "Content Details"])
    writer.writerow(["Target Market", target_market])
    writer.writerow(["Product Name", clean_for_csv(data.product_name)])
    writer.writerow(["Viability Score", f"{data.viability_score}/100"])
    writer.writerow(["Is Winning Product", str(data.is_winning_product)])
    writer.writerow(["Estimated Price Range", clean_for_csv(data.price_range)])
    writer.writerow(["Key Pros", " \n ".join([clean_for_csv(p) for p in data.pros])])
    writer.writerow(["Key Cons", " \n ".join([clean_for_csv(c) for c in data.cons])])
    writer.writerow(["Strategic Recommendation", clean_for_csv(data.recommendation)])
    writer.writerow(["Pricing Strategy", clean_for_csv(data.pricing_strategy)])
    writer.writerow(["Sourcing Blueprint", clean_for_csv(data.sourcing_info)])
    
    return output.getvalue().encode('utf-8-sig')


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
st.subheader("Multi-Agent Intelligence Platform & Winning Product Finder")

# Sidebar Configuration
st.sidebar.header("🔑 Credentials & Settings")
default_key = os.environ.get("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input("Groq API Key", value=default_key, type="password")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Target Market Selector
st.sidebar.markdown("---")