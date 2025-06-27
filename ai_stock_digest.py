import os
import sys
import requests
from datetime import datetime, timedelta
from collections import Counter
import openai
from textblob import TextBlob

# --- CONFIG ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
REDDIT_SUBS = ["stocks", "wallstreetbets"]
POST_LIMIT = 20
NEWS_LIMIT = 30

# --- LOGGING ---
def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

# --- SETUP ---
if not OPENAI_API_KEY:
    log("Missing OpenAI API key. Exiting.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

# --- YAHOO FINANCE ---
def get_yahoo_price_and_stats(ticker):
    url = f'https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}'
    try:
        r = requests.get(url, timeout=5)
        data = r.json()["quoteResponse"]["result"][0]
        price = data.get("regularMarketPrice", "N/A")
        change = data.get("regularMarketChangePercent", 0)
        cap = data.get("marketCap", "N/A")
        pe = data.get("trailingPE", "N/A")
        volume = data.get("regularMarketVolume", "N/A")
        return price, change, cap, pe, volume
    except Exception:
        return "N/A", 0, "N/A", "N/A", "N/A"

def get_yahoo_earnings_date(ticker):
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=calendarEvents"
    try:
        r = requests.get(url, timeout=5)
        earnings = r.json()["quoteSummary"]["result"][0]["calendarEvents"]["earnings"]["earningsDate"][0]["fmt"]
        return earnings
    except Exception:
        return "N/A"

# --- DATA SCRAPING (DUMMY FOR DEMO) ---
def get_all_us_tickers():
    # Replace with a real US tickers list if you want
    return ["AAPL", "TSLA", "NVDA", "MSFT", "GOOG"]

def build_ticker_lookup(ticker_list):
    return {t: [t.lower()] for t in ticker_list}

def scrape_reddit(subs=REDDIT_SUBS, limit=POST_LIMIT):
    # Dummy data for demo; replace with praw or pushshift as needed
    now = datetime.utcnow()
    data = []
    for t in ["AAPL", "TSLA", "NVDA", "MSFT", "GOOG"]:
        for i in range(limit):
            data.append({
                "ticker": t,
                "title": f"{t} Reddit post {i+1}",
                "text": f"Discussion about {t} number {i+1}",
                "url": f"https://reddit.com/r/stocks/{t.lower()}/post{i+1}",
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "score": 100 - i  # Simulate upvotes
            })
    return data

def scrape_news():
    # Dummy data for demo; replace with real news scraping
    data = []
    for t in ["AAPL", "TSLA", "NVDA", "MSFT", "GOOG"]:
        for i in range(NEWS_LIMIT):
            data.append({
                "ticker": t,
                "title": f"{t} news headline {i+1}",
                "url": f"https://news.com/{t.lower()}/news{i+1}",
                "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat()
            })
    return data

def get_top_tickers(reddit_posts, news_items):
    all_mentions = [p["ticker"] for p in reddit_posts] + [n["ticker"] for n in news_items]
    most_common = Counter(all_mentions).most_common(5)
    return [t[0] for t in most_common]

# --- SENTIMENT & HIGHLIGHTS ---
def analyze_sentiment(posts):
    sentiments = []
    for p in posts:
        tb = TextBlob(p['title'] + " " + p['text'])
        sentiments.append(tb.sentiment.polarity)
    if sentiments:
        avg = sum(sentiments) / len(sentiments)
        if avg > 0.15:
            return "positive"
        elif avg < -0.15:
            return "negative"
        else:
            return "neutral"
    return "neutral"

def get_news_sentiment(news_items, ticker):
    headlines = [n['title'] for n in news_items if n['ticker'] == ticker]
    if not headlines:
        return "neutral"
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    avg = sum(scores) / len(scores)
    if avg > 0.15:
        return "positive"
    elif avg < -0.15:
        return "negative"
    return "neutral"

def get_reddit_highlight(reddit_posts, ticker):
    filtered = [p for p in reddit_posts if p['ticker'] == ticker]
    if not filtered:
        return None
    return max(filtered, key=lambda p: p.get("score", 0))

def get_reddit_mentions_over_time(reddit_posts, ticker, days=7):
    now = datetime.utcnow()
    counts = [0]*days
    for p in reddit_posts:
        if p['ticker'] != ticker:
            continue
        ts = datetime.fromisoformat(p['timestamp'])
        day_delta = (now - ts).days
        if 0 <= day_delta < days:
            counts[days-1-day_delta] += 1
    return counts

def make_sparkline(data):
    if not data:
        return ""
    max_val = max(data)
    if max_val == 0:
        max_val = 1
    bars = ""
    w = 60
    h = 24
    bar_w = w // len(data)
    for i, val in enumerate(data):
        bar_h = int(h * val / max_val)
        y = h - bar_h
        bars += f'<rect x="{i*bar_w}" y="{y}" width="{bar_w-1}" height="{bar_h}" fill="#6cf0f7"/>'
    svg = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="vertical-align:middle">{bars}</svg>'
    return svg

def sentiment_emoji(sent):
    return {"positive":"🟢", "negative":"🔴", "neutral":"⚪"}[sent]

def human_number(n):
    try:
        n = float(n)
        if n >= 1e12:
            return f"${n/1e12:.2f}T"
        elif n >= 1e9:
            return f"${n/1e9:.2f}B"
        elif n >= 1e6:
            return f"${n/1e6:.2f}M"
        elif n >= 1e3:
            return f"${n/1e3:.2f}K"
        else:
            return str(n)
    except Exception:
        return str(n)

# --- SMART SUMMARIZATION ---
def summarize_news(ticker, news_items):
    news_texts = [n['title'] for n in news_items if n['ticker'] == ticker]
    if not news_texts:
        return "No recent news articles found."
    combined = "\n".join(news_texts)[:7000]
    prompt = f"""
You are a financial news analyst. Review the following recent news headlines about the stock ${ticker}.

1. Identify the main themes and news events.
2. Synthesize the overall media tone (bullish, bearish, or mixed).
3. Highlight any major catalysts, controversies, or risks mentioned.
4. Create a headline-style summary (max 20 words) as the first line.
5. Then provide a TL;DR with up to 3 bullet points.

News headlines:
{combined}
"""
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"News summary unavailable for {ticker}. Error: {e}"

def summarize_reddit(ticker, reddit_posts, sentiment):
    reddit_texts = [p['title'] + "\n" + p['text'] for p in reddit_posts if p['ticker'] == ticker]
    if not reddit_texts:
        return "No recent Reddit discussions found."
    combined = "\n\n".join(reddit_texts)[:7000]
    prompt = f"""
You are a financial social sentiment analyst. Analyze the following Reddit posts about ${ticker}.

1. Summarize the most common community opinions, highlighting both bull and bear arguments.
2. Identify any consensus, rumors, or new controversies.
3. Note if sentiment is shifting compared to the past week.
4. Give an overall sentiment assessment (positive, negative, or neutral), and explain why.
5. Write a headline-style summary (max 20 words) as the first line.
6. Finish with a TL;DR of up to 3 bullet points.

The general sentiment is: {sentiment}

Reddit posts:
{combined}
"""
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Reddit summary unavailable for {ticker}. Error: {e}"

# --- HTML GENERATION ---
def build_html(summaries, mention_counts, reddit_posts, news_items, last_updated=None):
    if last_updated is None:
        last_updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html_blocks = ""
    for ticker, content in summaries.items():
        count = mention_counts.get(ticker, 0)
        price, change, cap, pe, volume = get_yahoo_price_and_stats(ticker)
        change_col = "#65e19c" if change >= 0 else "#e68181"
        earnings = get_yahoo_earnings_date(ticker)
        news_sent = get_news_sentiment(news_items, ticker)
        highlight = get_reddit_highlight(reddit_posts, ticker)
        trend = get_reddit_mentions_over_time(reddit_posts, ticker)
        sparkline = make_sparkline(trend)
        badges = []
        if abs(change) > 5: badges.append("🚨 Volatile")
        if count > 20: badges.append("🔥 Trending")
        y_url = f"https://finance.yahoo.com/quote/{ticker}"
        tv_url = f"https://www.tradingview.com/symbols/{ticker}/"
        reddit_url = f"https://www.reddit.com/search/?q=%24{ticker}"
        html_blocks += f"""
        <div class="card">
            <div class="card-header">
                <span class="ticker">${ticker}</span>
                <span class="badges">{' '.join(badges)}</span>
            </div>
            <div class="card-row">
                <span class="price">${price if price!='N/A' else 'N/A'} <span style="color:{change_col};font-weight:600;">({change:+.2f}%)</span></span>
                <span class="sparkline">{sparkline}</span>
            </div>
            <div class="stats">
                <span>Market Cap: {human_number(cap)}</span>
                <span>P/E: {pe}</span>
                <span>Vol: {human_number(volume)}</span>
                <span>Earnings: {earnings}</span>
            </div>
            <div class="mentions">
                <span>Reddit mentions (7d): <b>{sum(trend)}</b> {sparkline}</span>
                <span>News sentiment: {sentiment_emoji(news_sent)} {news_sent.title()}</span>
            </div>
            <div class="section">
                <h3>📰 News Summary</h3>
                <div>{content['news']}</div>
            </div>
            <div class="section">
                <h3>🤖 Reddit Summary</h3>
                <div>{content['reddit']}</div>
            </div>
            <div class="links">
                <a href="{y_url}" target="_blank">Yahoo Finance</a> | 
                <a href="{tv_url}" target="_blank">TradingView</a> | 
                <a href="{reddit_url}" target="_blank">Reddit Search</a>
            </div>
            {"<div class='highlight'>Most upvoted Reddit post:<br><b>"+highlight['title']+"</b><br><a href='"+highlight['url']+"' target='_blank'>View post</a></div>" if highlight else ""}
        </div>
        """
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Stock Digest - Top 5</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    body {{
        background: #181c1f;
        color: #e6e9ef;
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, 'Liberation Sans', sans-serif;
        margin: 0;
        padding: 0;
    }}
    header {{
        background: linear-gradient(90deg, #232526 0%, #414345 100%);
        padding: 2rem 1rem 1rem 1rem;
        text-align: center;
        border-bottom: 2px solid #30363d;
    }}
    h1 {{
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }}
    .subtitle {{
        color: #9da5b4;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }}
    .timestamp {{
        color: #6d758d;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }}
    .container {{
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
        justify-content: center;
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }}
    .card {{
        background: #21262d;
        border-radius: 1.3rem;
        box-shadow: 0 6px 24px 0 #000c1c44;
        padding: 2rem 1.7rem 1.5rem 1.7rem;
        margin-bottom: 1.5rem;
        width: 390px;
        max-width: 97vw;
        border: 1.5px solid #30363d;
        transition: transform 0.15s;
    }}
    .card:hover {{
        transform: translateY(-7px) scale(1.022);
        box-shadow: 0 16px 32px 0 #000c1c55;
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #2c313a;
        padding-bottom: 0.2rem;
    }}
    .ticker {{
        font-size: 2.2rem;
        font-weight: 700;
        color: #6cf0f7;
        letter-spacing: 1px;
    }}
    .badges {{
        font-size: 1.09rem;
        font-weight: 600;
        color: #ffb85c;
        border-radius: 0.5rem;
        padding: 0.18rem 0.7rem;
        margin-left: 0.4rem;
    }}
    .card-row {{
        display: flex;
        align-items: center;
        gap: 1.2rem;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
    }}
    .price {{
        font-weight: 500;
        font-size: 1.2rem;
    }}
    .sparkline {{
        vertical-align: middle;
        margin-left: 0.3rem;
    }}
    .stats {{
        font-size: 0.97rem;
        color: #b0bac7;
        display: flex;
        flex-wrap: wrap;
        gap: 1.1rem;
        margin-bottom: 0.5rem;
    }}
    .mentions {{
        color: #ffb85c;
        font-size: 1.01rem;
        margin-bottom: 0.6rem;
        font-weight: 500;
        display: flex;
        gap: 2rem;
    }}
    .section {{
        margin-bottom: 1.1rem;
        padding-bottom: 0.3rem;
    }}
    h3 {{
        font-size: 1.17rem;
        margin-bottom: 0.5rem;
        font-weight: 550;
        color: #6cf0f7;
        letter-spacing: 0.5px;
    }}
    .links {{
        font-size: 1rem;
        margin-top: 0.8rem;
        color: #75e3ff;
    }}
    .links a, .links a:visited {{
        color: #75e3ff;
        text-decoration: underline;
        transition: color 0.2s;
    }}
    .links a:hover {{
        color: #e6e9ef;
    }}
    .highlight {{
        background: #2c2c36;
        border-radius: 0.6rem;
        margin-top: 0.6rem;
        padding: 0.7rem 1rem 0.4rem 1rem;
        color: #fff0c0;
        font-size: 0.98rem;
    }}
    footer {{
        text-align: center;
        font-size: 1.1rem;
        color: #6d758d;
        margin-bottom: 1.2rem;
        margin-top: 2.7rem;
    }}
    @media (max-width: 900px) {{
        .container {{
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
            padding: 1rem 0.5rem;
        }}
        .card {{
            width: 99vw;
            max-width: 100vw;
        }}
    }}
    </style>
</head>
<body>
    <header>
        <h1>AI Stock Digest</h1>
        <div class="subtitle">The hottest stocks on Reddit & in the news</div>
        <div class="timestamp">Last updated: {last_updated}</div>
    </header>
    <div class="container">
        {html_blocks}
    </div>
    <footer>
        <span>🚀 Powered by AI Digest &bull; <a href="https://github.com/AiStockDigest">GitHub</a></span>
        <br>
        <br>
        <strong>FAQ:</strong> This digest summarizes recent news and Reddit discussion for trending stocks. Sparkline shows 7-day Reddit mention trend. Sentiment is analyzed automatically; data may be incomplete or delayed. Use direct links for more info.
    </footer>
</body>
</html>
"""

# --- MAIN ---
def run_daily_digest():
    try:
        log("Starting Reddit scrape...")
        reddit_data = scrape_reddit()
        log("Starting news scrape...")
        news_data = scrape_news()
        log("Determining top 5 tickers...")
        top_5 = get_top_tickers(reddit_data, news_data)
        summaries = {}
        mention_counts = {ticker: sum(1 for p in reddit_data if p["ticker"] == ticker) for ticker in top_5}
        for ticker in top_5:
            sentiment = analyze_sentiment([p for p in reddit_data if p['ticker'] == ticker])
            news_summary = summarize_news(ticker, news_data)
            reddit_summary = summarize_reddit(ticker, reddit_data, sentiment)
            summaries[ticker] = {
                "news": news_summary,
                "reddit": reddit_summary
            }
        html = build_html(summaries, mention_counts, reddit_data, news_data)
        with open("daily_digest.html", "w", encoding="utf-8") as f:
            f.write(html)
        log("✔ AI Stock Digest written to daily_digest.html")
    except Exception as e:
        log(f"❗ Failed to generate digest: {e}")
        with open("daily_digest.html", "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Error creating digest</h1><p>{e}</p></body></html>")
        sys.exit(1)

if __name__ == "__main__":
    run_daily_digest()
