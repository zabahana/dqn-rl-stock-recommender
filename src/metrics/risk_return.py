from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_risk_return(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in price_data.items():
        if df.empty or "adj_close" not in df.columns:
            # yfinance download columns are lowercased with spaces replaced; sometimes 'Adj Close' -> 'adj_close'
            # fall back to 'close' if needed
            close_col = "adj_close" if "adj_close" in df.columns else "close"
        else:
            close_col = "adj_close"

        if close_col not in df.columns or df[close_col].isna().all():
            continue
        prices = df[close_col].dropna()
        if len(prices) < 30:
            continue
        returns = prices.pct_change().dropna()

        mean_daily = float(returns.mean())
        std_daily = float(returns.std(ddof=1))
        ann_factor = 252.0
        exp_return_annual = mean_daily * ann_factor
        vol_annual = std_daily * np.sqrt(ann_factor)
        sharpe = exp_return_annual / vol_annual if vol_annual > 0 else 0.0

        rows.append({
            "ticker": ticker,
            "exp_return_annual": exp_return_annual,
            "vol_annual": vol_annual,
            "sharpe": sharpe,
            "num_days": len(prices),
        })
    return pd.DataFrame(rows)


