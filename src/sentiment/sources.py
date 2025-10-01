from datetime import datetime, timedelta
from typing import Dict, List

import feedparser


def _rss_feeds_for_ticker(ticker: str) -> List[str]:
    query = ticker
    return [
        f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={query}+earnings&hl=en-US&gl=US&ceid=US:en",
    ]


def fetch_news_for_tickers(tickers: List[str], days: int = 30) -> Dict[str, List[Dict[str, str]]]:
    from datetime import timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result: Dict[str, List[Dict[str, str]]] = {t: [] for t in tickers}
    for t in tickers:
        for url in _rss_feeds_for_ticker(t):
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries:
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    if published and published >= cutoff:
                        result[t].append({
                            "title": getattr(entry, "title", ""),
                            "summary": getattr(entry, "summary", ""),
                            "link": getattr(entry, "link", ""),
                            "published": published.isoformat(),
                        })
            except Exception:
                continue
    return result


