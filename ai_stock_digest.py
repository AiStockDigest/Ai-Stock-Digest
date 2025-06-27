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
