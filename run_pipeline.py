import os
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.data.yahoo_fetch import fetch_history_for_tickers
from src.sentiment.sources import fetch_news_for_tickers
from src.sentiment.analyzer import score_sentiment
from src.metrics.risk_return import compute_risk_return
from src.metrics.projection import project_next_quarter
from src.visualization.plots import plot_price_history, plot_returns_distribution
from src.evaluation.report import build_recommendation_report
from src.analysis.eda import run_comprehensive_eda
from src.backtesting.backtester import run_comprehensive_backtesting
from src.optimization.hyperopt import run_hyperparameter_optimization


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main() -> None:
    cfg = DEFAULT_CONFIG
    ensure_dir(cfg.outputs_dir)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=cfg.lookback_days + 30)

    # 1) Market data
    price_data = fetch_history_for_tickers(cfg.tickers, start=start_date, end=end_date)
    print(f"Fetched price data for {len(price_data)} tickers: {list(price_data.keys())}")

    # 2) News + sentiment
    news_items = fetch_news_for_tickers(cfg.tickers, days=cfg.sentiment_window_days)
    sentiment_scores = score_sentiment(news_items)
    print(f"Computed sentiment scores: {sentiment_scores}")

    # 3) Risk/Return + projection
    metrics = compute_risk_return(price_data)
    print(f"Computed metrics for {len(metrics)} tickers")
    print(f"Metrics columns: {list(metrics.columns)}")
    projections = project_next_quarter(metrics, sentiment_scores)
    print(f"Generated projections for {len(projections)} tickers")

    # 4) Exploratory Data Analysis
    print("Running comprehensive exploratory data analysis...")
    eda_results = run_comprehensive_eda()
    print("EDA completed successfully.")

    # 5) Hyperparameter Optimization
    print("Running hyperparameter optimization...")
    hyperopt_results = run_hyperparameter_optimization(n_trials=30)
    print("Hyperparameter optimization completed successfully.")

    # 6) Backtesting Analysis
    print("Running comprehensive backtesting analysis...")
    backtest_results = run_comprehensive_backtesting()
    print("Backtesting analysis completed successfully.")

    # 7) Visualizations
    plot_price_history(price_data, os.path.join(cfg.outputs_dir, "price_history.png"))
    plot_returns_distribution(price_data, os.path.join(cfg.outputs_dir, "returns_distribution.png"))

    # 8) Recommendation report
    report_df, top5 = build_recommendation_report(
        metrics=metrics,
        projections=projections,
        sentiment_scores=sentiment_scores,
        out_dir=cfg.outputs_dir,
    )

    # Save artifacts
    report_df.to_csv(os.path.join(cfg.outputs_dir, "recommendations.csv"), index=False)
    with open(os.path.join(cfg.outputs_dir, "top5.json"), "w", encoding="utf-8") as f:
        json.dump(top5, f, indent=2)

    # Save analysis results
    with open(os.path.join(cfg.outputs_dir, "eda_results.json"), "w", encoding="utf-8") as f:
        json.dump(eda_results, f, indent=2, default=str)
    
    with open(os.path.join(cfg.outputs_dir, "hyperopt_results.json"), "w", encoding="utf-8") as f:
        json.dump(hyperopt_results, f, indent=2, default=str)
    
    with open(os.path.join(cfg.outputs_dir, "backtest_results.json"), "w", encoding="utf-8") as f:
        json.dump(backtest_results, f, indent=2, default=str)

    print("Enhanced pipeline complete. All artifacts saved under:", cfg.outputs_dir)
    print("Generated reports:")
    print("- EDA comprehensive report: eda_comprehensive_report.html")
    print("- Hyperparameter optimization report: hyperopt_report.html")
    print("- Backtesting reports for each strategy")
    print("- Performance analysis reports")


if __name__ == "__main__":
    main()


