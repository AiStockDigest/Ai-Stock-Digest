# AI Stock Digest: Fully Automated Top 5 Daily Summary Web App (Reddit + Twitter + News + Price/Sentiment Charts)

import praw
import re
import openai
import feedparser
import yfinance as yf
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from collections import Counter

# ⛔️ Monkey patch to ignore SSL errors for Twitter
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Reddit setup with read-only mode
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="stock_digest_scraper"
)
reddit.read_only = True

# OpenAI key from environment

# Dynamically fetch entire US stock market tickers from NASDAQ and NYSE listings
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

def build_ticker_lookup():
    tickers = get_all_us_tickers()
    lookup = {}
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        for _, row in sp500_table.iterrows():
            ticker = row['Symbol']
            name = row['Security']
            variations = set()
            variations.add(ticker.lower())
            variations.add(name.lower())
            variations.add(name.replace(" ", "").lower())
            variations.add(name.replace(" ", "").replace(".com", "").lower())
            variations.add(name.lower().replace("inc.", ""))
            variations.add(name.lower().replace("corp.", ""))
            variations.add(name.lower().replace("co.", ""))
            variations.add(name.lower().replace("inc", ""))
            variations.add(name.lower().replace("corporation", ""))
            variations.add(name.lower().split()[0])  # first word only
            lookup[ticker] = list(set(variations))
    except Exception as e:
        print("Error building lookup:", e)
        for ticker in tickers:
            lookup[ticker] = [ticker.lower()]
    return lookup

TICKER_LIST = get_all_us_tickers()
TICKER_LOOKUP = build_ticker_lookup()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Placeholder Twitter scraping function

def scrape_twitter(tickers, limit=50):
    print("Twitter scraping is disabled — using placeholder tweets.")
    tweets = []
    for ticker in tickers:
        tweets.append({
            "ticker": ticker,
            "text": f"Placeholder tweet about ${ticker} (Twitter data unavailable)",
            "user": "placeholder_bot",
            "followers": 999,
            "timestamp": datetime.utcnow().isoformat()
        })
    return tweets

# Define tickers to track
TICKER_LIST = ["AAPL", "TSLA", "NVDA", "AMD", "GOOG", "AMZN", "META", "MSFT", "GME", "NFLX", "BABA", "UBER", "DIS", "SPY"]

def extract_tickers(text):
    return re.findall(r'\$[A-Z]{1,5}', text)

def scrape_reddit(subs=["stocks", "wallstreetbets"], limit=50):
    posts = []
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=limit):
            tickers = extract_tickers(post.title + " " + post.selftext)
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

def scrape_twitter(tickers, limit=50):
    tweets = []
    for ticker in tickers:
        query = f"${ticker} lang:en"
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            if tweet.user.verified or tweet.user.followersCount > 500:
                tweets.append({
                    "ticker": ticker,
                    "text": tweet.content,
                    "user": tweet.user.username,
                    "followers": tweet.user.followersCount,
                    "timestamp": tweet.date.isoformat()
                })
    return tweets

def get_top_tickers(reddit_posts, news_items, tweet_items):
    all_mentions = [p["ticker"] for p in reddit_posts] + [n["ticker"] for n in news_items] + [t["ticker"] for t in tweet_items]
    most_common = Counter(all_mentions).most_common(5)
    return [t[0] for t in most_common]

def summarize_ticker(ticker, reddit_posts, news_items, tweet_items):
    reddit_texts = [p['title'] + "\n" + p['text'] for p in reddit_posts if p['ticker'] == ticker]
    news_texts = [n['title'] for n in news_items if n['ticker'] == ticker]
    tweet_texts = [t['text'] for t in tweet_items if t['ticker'] == ticker]
    combined = "\n\n".join(reddit_texts + news_texts + tweet_texts)[:7000]

    prompt = f"""
    Summarize all the Reddit posts, tweets, and news headlines below about ${ticker} in 2 clickbait-style paragraphs. Then include a TL;DR of 3 bullet points.

    Text:
    {combined}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def generate_price_and_volume_chart(ticker, reddit_posts, tweet_data):
    data = yf.download(ticker, period="5d", interval="1h")
    if data.empty:
        return None

    mention_times = [datetime.fromisoformat(p['timestamp']) for p in reddit_posts + tweet_data if p['ticker'] == ticker]
    hours = [t.replace(minute=0, second=0, microsecond=0) for t in mention_times]
    counts = Counter(hours)
    hourly_mentions = sorted(counts.items())
    mention_times_x = [x[0] for x in hourly_mentions]
    mention_counts_y = [x[1] for x in hourly_mentions]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title(f"{ticker} Price & Mentions (5 Days)")
    ax1.plot(data.index, data['Close'], color='tab:blue', label='Price')
    ax1.set_ylabel('Price ($)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.bar(mention_times_x, mention_counts_y, width=0.03, color='tab:red', alpha=0.6, label='Mentions')
    ax2.set_ylabel('Mentions', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    filename = f"{ticker}_chart.png"
    plt.savefig(filename)
    plt.close()
    return filename

def build_html(summaries):
    html_blocks = ""
    for ticker, content in summaries.items():
        chart_path = f"{ticker}_chart.png"
        img_tag = f'<img src="{chart_path}" width="700">' if os.path.exists(chart_path) else ""
        html_blocks += f"<h2>${ticker}</h2>{img_tag}<div>{content}</div><hr>"
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
    tweet_data = scrape_twitter(TICKER_LIST)
    top_5 = get_top_tickers(reddit_data, news_data, tweet_data)

    summaries = {}
    for ticker in top_5:
        summary = summarize_ticker(ticker, reddit_data, news_data, tweet_data)
        generate_price_and_volume_chart(ticker, reddit_data, tweet_data)
        summaries[ticker] = summary

    html = build_html(summaries)
    with open("daily_digest.html", "w") as f:
        f.write(html)
    print("✔ AI Stock Digest written to daily_digest.html")

if __name__ == "__main__":
    run_daily_digest()
