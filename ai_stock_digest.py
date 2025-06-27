# AI Stock Digest: Fully Automated Top 5 Daily Summary Web App (Reddit + News + Price/Sentiment Charts)

import praw
import re
import openai
import feedparser
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import pickle

# Reddit setup with read-only mode
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="stock_digest_scraper"
)
reddit.read_only = True

# OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Try fuzzywuzzy
try:
    from fuzzywuzzy import fuzz
except ImportError:
    print("fuzzywuzzy is not installed. Fuzzy matching will be disabled.")
    fuzz = None

# Get all tickers from NASDAQ/NYSE (fallback to S&P + Russell)
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
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        russell = pd.read_html('https://en.wikipedia.org/wiki/Russell_2000_Index')[2]['Ticker'].tolist()
        return list(set(sp500 + russell))

# Build lookup dict for ticker name variants
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
            if fuzz and fuzz.ratio(input_str, variant) >= threshold:
                return ticker
    return None

# Get trending tickers from Reddit
def get_top_discussed_tickers(limit=10, subs=["stocks", "wallstreetbets"], hours=24):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    ticker_counter = Counter()
    for sub in subs:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.new(limit=500):
            if datetime.utcfromtimestamp(post.created_utc) < start_time:
                continue
            tickers = re.findall(r'\$[A-Z]{1,5}', post.title + " " + post.selftext)
            ticker_counter.update([t.strip("$") for t in tickers])
    return [ticker for ticker, _ in ticker_counter.most_common(limit)]

TICKER_LIST = get_top_discussed_tickers()
TICKER_LOOKUP = build_ticker_lookup()
