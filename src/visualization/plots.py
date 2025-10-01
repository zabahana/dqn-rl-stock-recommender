from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_price_history(price_data: Dict[str, pd.DataFrame], out_path: str) -> None:
    plt.figure(figsize=(12, 6))
    has_data = False
    for ticker, df in price_data.items():
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        if close_col in df.columns and not df[close_col].empty:
            df[close_col].plot(label=ticker)
            has_data = True
    
    if has_data:
        plt.title("Price History")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
    else:
        plt.title("No Price Data Available")
        plt.text(0.5, 0.5, "No valid price data found", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_returns_distribution(price_data: Dict[str, pd.DataFrame], out_path: str) -> None:
    plt.figure(figsize=(12, 6))
    has_data = False
    for ticker, df in price_data.items():
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        if close_col in df.columns and not df[close_col].empty:
            returns = df[close_col].pct_change().dropna()
            if not returns.empty:
                sns.kdeplot(returns, label=ticker, fill=True, alpha=0.15)
                has_data = True
    
    if has_data:
        plt.title("Daily Returns Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Density")
        plt.legend()
    else:
        plt.title("No Returns Data Available")
        plt.text(0.5, 0.5, "No valid returns data found", ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


