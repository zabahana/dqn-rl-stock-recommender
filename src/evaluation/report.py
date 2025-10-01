import os
from typing import Dict, List, Tuple

import pandas as pd


def build_recommendation_report(
    metrics: pd.DataFrame,
    projections: pd.DataFrame,
    sentiment_scores: Dict[str, float],
    out_dir: str,
) -> Tuple[pd.DataFrame, List[str]]:
    # Handle empty DataFrames
    if projections.empty:
        # Create a minimal report with sentiment scores only
        df = pd.DataFrame([{"ticker": ticker, "sentiment": score} for ticker, score in sentiment_scores.items()])
        df["proj_return_quarter"] = 0.0
        df["proj_risk_quarter"] = 0.0
        df["proj_sharpe_quarter"] = 0.0
        df["score"] = df["sentiment"]  # Use sentiment as score when no other data
    else:
        # Check if metrics has the required columns
        if "sharpe" in metrics.columns and not metrics.empty:
            df = projections.merge(metrics[["ticker", "sharpe"]], on="ticker", how="left", suffixes=("", "_hist"))
        else:
            # If sharpe column doesn't exist, just use projections
            df = projections.copy()
            df["sharpe_hist"] = 0.0  # Default value
        
        df["sentiment"] = df["ticker"].map(sentiment_scores).fillna(0.0)

    # Composite score: prefer high projected Sharpe, return; penalize risk; adjust by sentiment
    df["score"] = (
        0.5 * df["proj_sharpe_quarter"] + 0.4 * df["proj_return_quarter"] - 0.2 * df["proj_risk_quarter"] + 0.2 * df["sentiment"]
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Select top 5 interesting
    top5 = df.head(5)["ticker"].tolist()

    # Save a lightweight HTML summary
    html_path = os.path.join(out_dir, "summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Next-Quarter Recommendations</title></head><body>")
        f.write("<h2>Next-Quarter Recommendations (Top 5)</h2>")
        f.write("<ul>")
        for t in top5:
            row = df[df["ticker"] == t].iloc[0]
            f.write(
                f"<li><b>{t}</b>: proj_return_q={row['proj_return_quarter']:.2%}, proj_risk_q={row['proj_risk_quarter']:.2%}, sentiment={row['sentiment']:.2f}, score={row['score']:.3f}</li>"
            )
        f.write("</ul>")
        f.write("<h3>Charts</h3>")
        f.write("<img src='price_history.png' style='max-width: 900px; display:block; margin-bottom: 16px;' />")
        f.write("<img src='returns_distribution.png' style='max-width: 900px; display:block; margin-bottom: 16px;' />")
        f.write("</body></html>")

    # Prepare CSV-friendly output
    report_cols = [
        "ticker",
        "sentiment",
        "proj_return_quarter",
        "proj_risk_quarter",
        "proj_sharpe_quarter",
        "score",
    ]
    out_df = df[report_cols].copy()
    return out_df, top5


