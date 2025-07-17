import requests

# === Replace with your own Newsdata.io API key ===
NEWSDATA_API_KEY = "YOUR-API-KEY"

# =============================
# Fetch Latest News for a Stock
# =============================
def fetch_news(ticker, count=5):
    ticker_query_map = {
        "^NSEI": "Nifty 50",
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "AAPL": "Apple Inc",
        "TSLA": "Tesla",
        "AMZN": "Amazon",
        "BTC-USD": "Bitcoin"
    }
    query = ticker_query_map.get(ticker, ticker)
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language=en"

    print("🛰️ News API URL:", url)  # For debugging

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get("results", [])[:count]
        if not articles:
            print("⚠️ No articles found.")
        return [f"{a['title']} - {a.get('description', '')}" for a in articles]
    except Exception as e:
        print("❌ Error:", e)
        return [f"Error fetching news: {e}"]
# =============================
# Summarize Using LLM
# =============================
def summarize_news(news_list, call_llm_fn):
    news_text = "\n".join(news_list)
    prompt = (
        "You are a financial analyst.\n"
        "Summarize the following stock-related news in bullet points. "
        "Include sentiment (positive/negative/neutral) and possible impact on stock price:\n\n"
        f"{news_text}"
    )
    return call_llm_fn(prompt)

