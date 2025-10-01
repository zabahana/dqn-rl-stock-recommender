from typing import Dict

import numpy as np
import pandas as pd


def project_next_quarter(metrics: pd.DataFrame, sentiment: Dict[str, float]) -> pd.DataFrame:
    if metrics.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["ticker", "proj_return_annual", "proj_vol_annual", 
                                   "proj_return_quarter", "proj_risk_quarter", "proj_sharpe_quarter"])
    
    df = metrics.copy()
    # Simple heuristic: sentiment scales expected return; mild penalty on volatility for negative sentiment
    s_map = sentiment
    scale_return = []
    adj_vol = []
    for _, row in df.iterrows():
        s = float(s_map.get(row["ticker"], 0.0))  # -1..1
        ret = float(row["exp_return_annual"]) * (1.0 + 0.3 * s)
        vol = float(row["vol_annual"]) * (1.0 + (-0.1 * s))
        scale_return.append(ret)
        adj_vol.append(max(vol, 1e-6))
    df["proj_return_annual"] = scale_return
    df["proj_vol_annual"] = adj_vol

    # Convert to next quarter (approx 63 trading days ~ 1/4 year)
    quarter_fraction = 63.0 / 252.0
    df["proj_return_quarter"] = df["proj_return_annual"] * quarter_fraction
    df["proj_risk_quarter"] = df["proj_vol_annual"] * np.sqrt(quarter_fraction)
    df["proj_sharpe_quarter"] = df["proj_return_quarter"] / df["proj_risk_quarter"].replace(0, 1e-6)
    return df


