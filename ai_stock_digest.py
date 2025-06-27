import praw
import re
import openai
import feedparser
import os
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import pickle
import sys

def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-16k")  # Default to gpt-3.5-turbo

if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    log("Missing Reddit API credentials. Exiting.")
    sys.exit(1)

if not OPENAI_API_KEY:
    log("Missing OpenAI API key. Exiting.")
    sys.exit(1)

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent="stock_digest_scraper"
)
reddit.read_only = True

openai.api_key = OPENAI_API_KEY

def get_all_us_tickers():
    nasdaq_url = 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download'
    nyse_url = 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download'
    cache_file = "all_us_tickers_cache.pkl"
    if os.path.exists(cache_file):
        log("Loading cached ticker list.")
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        log("Downloading ticker lists...")
        nasdaq_df = pd.read_csv(nasdaq_url, timeout=15)
        nyse_df = pd.read_csv(nyse_url, timeout=15)
        combined_df = pd.concat([nasdaq_df, nyse_df])
        tickers = combined_df['Symbol'].dropna().unique().tolist()
        if tickers:
            with open(cache_file, "wb") as f:
                pickle.dump(tickers, f)
        log(f"Loaded {len(tickers)} tickers from exchanges.")
        return tickers
    except Exception as e:
        log(f"Error fetching ticker list: {e}")
        return ["AAPL", "GOOG", "TSLA", "MSFT", "NVDA"]

try:
    from fuzzywuzzy import fuzz
except ImportError:
    log("fuzzywuzzy not installed. Fuzzy matching disabled.")
    fuzz = None

def build_ticker_lookup(ticker_list):
    cache_file = "ticker_lookup_cache.pkl"
    if os.path.exists(cache_file):
        log("Loading cached ticker lookup.")
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    lookup = {}
    try:
        log("Building ticker lookup from Wikipedia S&P 500...")
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', timeout=20)[0]
        for _, row in sp500_table.iterrows():
            ticker = row['Symbol']
            name = row['Security']
            if ticker not in ticker_list:
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
        log(f"Error building lookup: {e}")
        for ticker in ticker_list:
            lookup[ticker] = [ticker.lower()]
    with open(cache_file, "wb") as f:
        pickle.dump(lookup, f)
    return lookup

def fuzzy_match(input_str, lookup_dict, threshold=80):
    if fuzz is None:
        return None
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
        try:
            for post in subreddit.new(limit=100):
                if datetime.utcfromtimestamp(post.created_utc) < start_time:
                    continue
                tickers = re.findall(r'\$[A-Z]{1,5}', post.title + " " + post.selftext)
                ticker_counter.update([t.strip("$") for t in tickers])
        except Exception as e:
            log(f"Error scraping subreddit {sub}: {e}")
    return [ticker for ticker, _ in ticker_counter.most_common(limit)]

TICKER_LIST = get_all_us_tickers()
TICKER_LOOKUP = build_ticker_lookup(TICKER_LIST)

def scrape_reddit(subs=["stocks", "wallstreetbets"], limit=20):
    posts = []
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        try:
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
        except Exception as e:
            log(f"Error scraping subreddit {sub}: {e}")
    log(f"Scraped {len(posts)} Reddit posts.")
    return posts

def scrape_news():
    feeds = []
    base_url = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    top_discussed = get_top_discussed_tickers(limit=5)
    for ticker in top_discussed:
        url = base_url.format(ticker=ticker)
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:2]:
                feeds.append({
                    "ticker": ticker,
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.published
                })
        except Exception as e:
            log(f"Error fetching news for {ticker}: {e}")
    log(f"Scraped {len(feeds)} news items.")
    return feeds

def get_top_tickers(reddit_posts, news_items):
    all_mentions = [p["ticker"] for p in reddit_posts] + [n["ticker"] for n in news_items]
    most_common = Counter(all_mentions).most_common(5)
    return [t[0] for t in most_common]

def summarize_ticker(ticker, reddit_posts, news_items):
    reddit_texts = [p['title'] + "\n" + p['text'] for p in reddit_posts if p['ticker'] == ticker]
    news_texts = [n['title'] for n in news_items if n['ticker'] == ticker]
    combined = "\n\n".join(reddit_texts + news_texts)[:7000]
    prompt = f"Summarize all the Reddit posts and news headlines below about ${ticker} in 2 clickbait-style paragraphs. Then include a TL;DR of 3 bullet points.\n\nText:\n{combined}"
    try:
        log(f"Summarizing {ticker} with model {OPENAI_MODEL}...")
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        log(f"Error summarizing {ticker}: {e}")
        return f"Summary unavailable for {ticker}. Error: {e}"

def build_html(summaries):
    html_blocks = ""
    for ticker, content in summaries.items():
        html_blocks += f"<h2>${ticker}</h2><div>{content}</div><hr>"
    return (
        "<html>"
        "<head><title>AI Stock Digest - Top 5</title></head>"
        "<body>"
        "<h1>Top 5 Stocks Today</h1>"
        f"{html_blocks}"
        "<footer><br><i>Updated daily by AI Digest</i></footer>"
        "</body>"
        "</html>"
    )

def run_daily_digest():
    log("Starting Reddit scrape...")
    reddit_data = scrape_reddit()
    log("Starting news scrape...")
    news_data = scrape_news()
    log("Determining top 5 tickers...")
    top_5 = get_top_tickers(reddit_data, news_data)
    log(f"Top 5: {top_5}")
    summaries = {}
    for ticker in top_5:
        summaries[ticker] = summarize_ticker(ticker, reddit_data, news_data)
    html = build_html(summaries)
    with open("daily_digest.html", "w", encoding="utf-8") as f:
        f.write(html)
    log("✔ AI Stock Digest written to daily_digest.html")

if __name__ == "__main__":
    run_daily_digest()
