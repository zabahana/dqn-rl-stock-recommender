from dataclasses import dataclass
from typing import List


@dataclass
class ProjectConfig:
    tickers: List[str]
    lookback_days: int = 252
    sentiment_window_days: int = 30
    outputs_dir: str = "outputs"
    random_seed: int = 42


DEFAULT_CONFIG = ProjectConfig(
    tickers=[
        # Original 7 technology stocks
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        # Additional 10 stocks for expanded universe
        "NFLX",  # Netflix - Streaming/Entertainment
        "CRM",   # Salesforce - Cloud/SaaS
        "ADBE",  # Adobe - Software/Creative
        "INTC",  # Intel - Semiconductors
        "AMD",   # Advanced Micro Devices - Semiconductors
        "PYPL",  # PayPal - Fintech/Payments
        "UBER",  # Uber - Transportation/Tech
        "SQ",    # Block (Square) - Fintech/Payments
        "ZM",    # Zoom - Video Communications
        "DOCU"   # DocuSign - Digital Documents
    ],
)


