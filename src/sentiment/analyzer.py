from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def score_sentiment(news_items: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
    analyzer = SentimentIntensityAnalyzer()
    ticker_to_score: Dict[str, float] = {}
    for ticker, items in news_items.items():
        scores = []
        for item in items:
            text = f"{item.get('title','')} {item.get('summary','')}"
            vs = analyzer.polarity_scores(text)
            scores.append(vs["compound"])  # -1..1
        ticker_to_score[ticker] = float(sum(scores) / len(scores)) if scores else 0.0
    return ticker_to_score


