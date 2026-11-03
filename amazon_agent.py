import os
import json
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# 1. Expanded Pydantic Model including pricing strategy
class MarketAnalysis(BaseModel):
    product_name: str = Field(description="Name or category of the product analyzed")
    price_range: str = Field(description="Estimated market price range, e.g., '$25 - $50'")
    pros: List[str] = Field(description="Top 3 key advantages or standout features from reviews")
    cons: List[str] = Field(description="Top 3 common drawbacks or customer complaints from reviews")
    recommendation: str = Field(description="Strategic advice based on review feedback and market positioning")
    pricing_strategy: str = Field(default="Pending analysis", description="Competitive pricing recommendations and margin strategy")

# 2. LangGraph Shared State
class AgentState(TypedDict):
    query: str
    raw_reviews: List[str]
    analysis: MarketAnalysis

# --- NODE 1: Live Web Scraper using BeautifulSoup ---
def review_scraper_node(state: AgentState):
    query = state["query"]
    print(f"🔍 [Node 1] Scraping live Amazon search data for: '{query}'...")
    
    formatted_query = query.replace(" ", "+")
    url = f"https://www.amazon.com/s?k={formatted_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US, en;q=0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    
    extracted_snippets = []
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("div", attrs={"data-component-type": "s-search-result"})
            
            for r in results[:5]:
                title_elem = r.select_one(".a-size-medium.a-color-base.a-text-normal")
                price_elem = r.select_one(".a-price .a-offscreen")
                rating_elem = r.select_one(".a-icon-star-small .a-icon-alt")
                
                title = title_elem.text.strip() if title_elem else "Unknown Product"
                price = price_elem.text.strip() if price_elem else "Price not listed"
                rating = rating_elem.text.strip() if rating_elem else "No rating"
                
                snippet = f"Product: {title} | Price: {price} | Rating: {rating}"
                extracted_snippets.append(snippet)
        
        if not extracted_snippets:
            raise Exception("Fallback triggered")
            
    except Exception:
        extracted_snippets = [
            f"Fallback Review 1 for {query}: High demand online, excellent build quality reported by users.",
            f"Fallback Review 2 for {query}: Pricing is competitive relative to market alternatives, though shipping delays were noted.",
            f"Fallback Review 3 for {query}: Outstanding core functionality with minor software/usability learning curves."
        ]
        
    return {"raw_reviews": extracted_snippets}

# --- NODE 2: Analyze Reviews & Format via Pydantic ---
def market_analyzer_node(state: AgentState):
    print("📊 [Node 2] Analyzing review data and generating structured Pydantic report...")
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    reviews_formatted = "\n".join([f"- {r}" for r in state["raw_reviews"]])
    
    system_prompt = (
        "You are an expert Amazon market research assistant. "
        "Analyze the product query and scraped product/review data, then respond strictly as a valid JSON object "
        "matching these keys: 'product_name', 'price_range', 'pros', 'cons', 'recommendation'. "
        "Note: Leave 'pricing_strategy' blank or omit it for now."
    )
    
    user_prompt = f"Product Query: {state['query']}\n\nScraped Market Data:\n{reviews_formatted}"
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    raw_json = response.choices[0].message.content
    structured_output = MarketAnalysis.model_validate_json(raw_json)
    
    return {"analysis": structured_output}

# --- NODE 3: Automated Pricing & Competitive Strategist ---
def pricing_strategist_node(state: AgentState):
    print("💰 [Node 3] Running Pricing & Competitive Strategist analysis...")
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    current_analysis = state["analysis"]
    
    system_prompt = (
        "You are an expert E-commerce Pricing and Margin Strategist. "
        "Review the product market analysis below and provide an actionable, concise competitive pricing strategy, "
        "recommended discount thresholds, and margin positioning advice."
    )
    
    user_prompt = f"""
    Product: {current_analysis.product_name}
    Estimated Price Range: {current_analysis.price_range}
    Pros: {', '.join(current_analysis.pros)}
    Cons: {', '.join(current_analysis.cons)}
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    strategy_text = response.choices[0].message.content
    
    # Mutate / update the Pydantic analysis object with the pricing strategy
    current_analysis.pricing_strategy = strategy_text
    
    return {"analysis": current_analysis}

# 3. Wire up the 3-Node Multi-Agent Graph Pipeline
workflow = StateGraph(AgentState)
workflow.add_node("review_scraper", review_scraper_node)
workflow.add_node("market_analyzer", market_analyzer_node)
workflow.add_node("pricing_strategist", pricing_strategist_node)

# Sequential flow: START -> review_scraper -> market_analyzer -> pricing_strategist -> END
workflow.add_edge(START, "review_scraper")
workflow.add_edge("review_scraper", "market_analyzer")
workflow.add_edge("market_analyzer", "pricing_strategist")
workflow.add_edge("pricing_strategist", END)

app = workflow.compile()