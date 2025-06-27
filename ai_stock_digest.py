# AI Stock Digest: Fully Automated Top 5 Daily Summary Web App (Reddit + Twitter + News + Price/Sentiment Charts)

import praw
import re
import openai
import feedparser
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from collections import Counter


# Reddit setup with read-only mode
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="stock_digest_scraper"
)
reddit.read_only = True

# OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

import pandas as pd

def get_all_us_tickers():
    nasdaq_url = 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'
    nyse_url = 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'
    try:
        nasdaq_df = pd.read_csv(nasdaq_url)
        nyse_df = pd.read_csv(nyse_url)
        combined_df = pd.concat([nasdaq_df, nyse_df])
        tickers = combined_df['Symbol'].dropna().unique().tolist()
        return tickers
    except Exception as e:
        print("Error fetching ticker list:", e)
                # Expanded fallback: S&P 500 + Russell 2000 tickers
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        russell = pd.read_html('https://en.wikipedia.org/wiki/Russell_2000_Index')[2]['Ticker'].tolist()
        return list(set(sp500 + russell))

try:
    from fuzzywuzzy import fuzz
except ImportError:
    print("fuzzywuzzy is not installed. Fuzzy matching will be disabled.")
    fuzz = None
import pickle

def build_ticker_lookup():
    cache_file = "ticker_lookup_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    lookup = {}
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        for _, row in sp500_table.iterrows():
            ticker = row['Symbol']
            name = row['Security']
            if ticker not in TICKER_LIST:
                continue
            variations = set()
            base = name.lower()
            variations.update([
                ticker.lower(),
                base,
                base.replace(" ", ""),
                base.replace(".com", ""),
                base.replace("inc.", ""),
                base.replace("corp.", ""),
                base.replace("co.", ""),
                base.replace("inc", ""),
                base.replace("corporation", ""),
                base.split()[0]
            ])
            lookup[ticker] = list(variations)
    except Exception as e:
        print("Error building lookup:", e)
        for ticker in TICKER_LIST:
            lookup[ticker] = [ticker.lower()]

    with open(cache_file, "wb") as f:
        pickle.dump(lookup, f)

    return lookup

# Optional fuzzy match utility
def fuzzy_match(input_str, lookup_dict, threshold=80):
    input_str = input_str.lower()
    for ticker, variants in lookup_dict.items():
        for variant in variants:
            if fuzz.ratio(input_str, variant) >= threshold:
                return ticker
    return None

def get_top_discussed_tickers(limit=10, subs=["stocks", "wallstreetbets"], hours=24):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    ticker_counter = Counter()
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.new(limit=500):  # Increase or decrease this depending on performance
            if datetime.utcfromtimestamp(post.created_utc) < start_time:
                continue
            tickers = re.findall(r'\$[A-Z]{1,5}', post.title + " " + post.selftext)
            ticker_counter.update([t.strip("$") for t in tickers])
    return [ticker for ticker, _ in ticker_counter.most_common(limit)]

TICKER_LIST = get_top_discussed_tickers()
TICKER_LOOKUP = build_ticker_lookup()


def scrape_reddit(subs=["stocks", "wallstreetbets"], limit=50):
    posts = []
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=limit):
            tickers = re.findall(r'\$[A-Z]{1,5}', post.title + " " + post.selftext)
            for t in tickers:
                posts.append({
                    "ticker": t.strip("$"),
                    "title": post.title,
                    "text": post.selftext,
                    "url": post.url,
                    "timestamp": datetime.utcfromtimestamp(post.created_utc).isoformat()
                })
    return posts

def scrape_news():
    feeds = []
    base_url = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    for ticker in TICKER_LIST:
        url = base_url.format(ticker=ticker)
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:5]:
            feeds.append({
                "ticker": ticker,
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })
    return feeds

def get_top_tickers(reddit_posts, news_items):
    all_mentions = [p["ticker"] for p in reddit_posts] + [n["ticker"] for n in news_items]
    most_common = Counter(all_mentions).most_common(5)
    return [t[0] for t in most_common]

def summarize_ticker(ticker, reddit_posts, news_items):
    reddit_texts = [p['title'] + "
" + p['text'] for p in reddit_posts if p['ticker'] == ticker]
    news_texts = [n['title'] for n in news_items if n['ticker'] == ticker]
    combined = "

".join(reddit_texts + news_texts)[:7000]

    prompt = f"""
    Summarize all the Reddit posts and news headlines below about ${ticker} in 2 clickbait-style paragraphs. Then include a TL;DR of 3 bullet points.

    Text:
    {combined}
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

".join(reddit_texts + news_texts)[:7000]

    prompt = f"""
    Summarize all the Reddit posts and news headlines below about ${ticker} in 2 clickbait-style paragraphs. Then include a TL;DR of 3 bullet points.

    Text:
    {combined}
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def build_html(summaries):
    html_blocks = ""
    for ticker, content in summaries.items():
        html_blocks += f"<h2>${ticker}</h2><div>{content}</div><hr>"
    return f"""
    <html>
    <head><title>AI Stock Digest - Top 5</title></head>
    <body>
    <h1>🔥 Top 5 Stocks Today</h1>
    {html_blocks}
    <footer><br><i>Updated daily by AI Digest</i></footer>
    </body>
    </html>
    """

def run_daily_digest():
    reddit_data = scrape_reddit()
    news_data = scrape_news()
    top_5 = get_top_tickers(reddit_data, news_data)

    summaries = {}
    for ticker in top_5:
        summaries[ticker] = summarize_ticker(ticker, reddit_data, news_data)

    html = build_html(summaries)
    with open("daily_digest.html", "w") as f:
        f.write(html)
    print("✔ AI Stock Digest written to daily_digest.html")


if __name__ == "__main__":
    run_daily_digest()


