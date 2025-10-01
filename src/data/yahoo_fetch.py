from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


def fetch_history_for_tickers(
    tickers: List[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if not df.empty:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.rename(columns={c: c.replace(" ", "_").lower() for c in df.columns})
                data[ticker] = df
                print(f"Successfully fetched {len(df)} rows for {ticker}")
            else:
                print(f"No data returned for {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue
    return data


